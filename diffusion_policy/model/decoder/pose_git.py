import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.invariant_pose_transformer import InvariantPoseTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention, InvariantPoseAttention
from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class FlowMatchingInvariantPointTransformer(nn.Module):
    def __init__(self,
                 obs_dim,
                 n_obs_steps,
                 n_action_steps,
                 latent_dim=128,
                 depth=2,
                 heads=3, dim_head=64,
                 kv_dim=None,
                 dropout=0.,
                 attention_module=InvariantPointAttention,
                 gripper_out = False,
                 ignore_collisions_out = False,
                 causal_attn=True,
                 point_dim=4,
                 **kwargs
                 ):
        super(FlowMatchingInvariantPointTransformer, self).__init__()

        ## Parameters ##
        self.n_obs_steps = n_obs_steps
        self.obs_dim = obs_dim
        self.n_action_steps = n_action_steps
        self.latent_dim = latent_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.kv_dim = kv_dim
        self.dropout = dropout
        self.attention_module = attention_module
        self.gripper_out = gripper_out
        self.ignore_collisions_out = ignore_collisions_out
        self.causal_attn = causal_attn

        ## Set Action Poses ##
        self.action_features = nn.Parameter(torch.randn(n_action_steps, latent_dim))
        self.positional_encoding = PositionalEncoding(latent_dim, max_len=n_action_steps)


        # ## Set Observation Encoder ##
        # self.obs_encoder =  nn.Linear(obs_dim, latent_dim)
        #
        # ## Set Time Encoder ##
        # self.time_encoder = SinusoidalPosEmb(latent_dim)

        ## Feature Encoder ##
        # self.encoder = nn.Sequential(
        #     nn.Linear(latent_dim*2, 8 * latent_dim),
        #     nn.Mish(),
        #     nn.Linear(8 * latent_dim, latent_dim)
        # )


        ## Set Observation Encoder ##
        self.obs_encoder =  nn.Sequential(
            nn.Linear(obs_dim, latent_dim*4),
            nn.Mish(),
            nn.Linear(latent_dim*4, latent_dim))


        ## Set Time Encoder ##
        self.time_encoder = nn.Sequential(
                    SinusoidalPosEmb(latent_dim),
                    nn.Linear(latent_dim, 4 * latent_dim),
                    nn.Mish(),
                    nn.LayerNorm(4 * latent_dim),
                    nn.Linear(4 * latent_dim, latent_dim))

        self.sigmoid = nn.Sigmoid()

        self.model = InvariantPointTransformer(latent_dim*2, depth, heads, dim_head, kv_dim=kv_dim,
                                          dropout=dropout, attention_module=attention_module, point_dim=point_dim, **kwargs)

        # self.model = InvariantPoseTransformer(latent_dim, depth, heads, dim_head, kv_dim=kv_dim,
        #                                       dropout=dropout, attention_module=InvariantPoseAttention, **kwargs)

        ## Set Output Layer ##
        out_dim = 6
        if self.gripper_out:
            out_dim += 1
        if self.ignore_collisions_out:
            out_dim += 1
        self.to_out = nn.Linear(latent_dim*2, out_dim)

    def set_context(self, context):
        self.context = context
        self.context['latent_features'] = self.obs_encoder(context['obs_f'])

    def forward(self, act_r, act_p, time):

        ## 1. Compute Time Embeddins ##
        time_emb = self.time_encoder(time)

        ## Set Action Features ##
        action_features = self.action_features[None, :, :].expand(act_r.shape[0], -1, -1)
        action_features = self.positional_encoding(action_features)

        ## Set Observation Features ##
        observation_features = self.context['latent_features']

        ## Build Features for Self-Attention Transformer and add time embedding##
        features = torch.cat((action_features, observation_features), dim=1) #+ time_emb[:, None, :]
        features = torch.cat((features, time_emb[:,None,:].repeat(1,features.shape[1],1)), dim=-1)
        #features = self.encoder(features)

        ## Set Joint Poses ##
        poses = {}

        poses['rotations'] = torch.cat((act_r, self.context['obs_r']), dim=1)
        poses['translations'] = torch.cat((act_p, self.context['obs_p']), dim=1)

        ## Set Mask ##
        act_tokens = act_r.shape[1]
        obs_tokens = self.context['obs_r'].shape[1]
        mask = torch.ones(act_tokens + obs_tokens, act_tokens + obs_tokens)
        if self.causal_attn:
            mask[:act_tokens, :act_tokens] = torch.tril(mask[:act_tokens, :act_tokens])
            mask[act_tokens:, :act_tokens] = torch.zeros_like(mask[act_tokens:, :act_tokens])
        # omit_object_attn = True
        # if omit_object_attn:
        #     n_gripper_obs = self.n_obs_steps
        #     mask[act_tokens+n_gripper_obs:, act_tokens+n_gripper_obs:] = torch.zeros_like(mask[act_tokens+n_gripper_obs:, act_tokens+n_gripper_obs:])

        mask = mask.to(features.device).bool()

        features_out = self.model(features, poses, mask=mask)

        out = self.to_out(features_out)

        out_dict = {
            "v": out[:, :act_tokens, :6],
            "act_f": features_out[:, :act_tokens, :],
            "obs_f": features_out[:, act_tokens:, :],

        }
        if self.gripper_out and self.ignore_collisions_out:
            out_dict["gripper"] = self.sigmoid(out[:, :act_tokens, 6])
            out_dict['ignore_collisions'] = self.sigmoid(out[:, :act_tokens, 7])
        if self.gripper_out:
            out_dict["gripper"] = self.sigmoid(out[:, :act_tokens, 6])
        elif self.ignore_collisions_out:
            out_dict['ignore_collisions'] = self.sigmoid(out[:, :act_tokens, 6])
        
        return out_dict
    
    def get_args(self):
        return {
            '__class__': [type(self).__module__, type(self).__name__],
            'params':{
                'obs_dim': self.obs_dim,
                'n_action_steps': self.n_action_steps,
                'latent_dim': self.latent_dim,
                'depth': self.depth,
                'heads': self.heads,
                'dim_head': self.dim_head,
                'kv_dim': self.kv_dim,
                'dropout': self.dropout,
                'attention_module': self.attention_module}
        }



def test():
    import time

    n_action_steps = 12
    obs_dim = 50
    context_dim = 64
    depth = 5
    heads = 3
    dim_head = 60

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FlowMatchingInvariantPointTransformer(obs_dim=obs_dim,
                                                  n_action_steps=n_action_steps,).to(device)

    ### RUN ###
    ## Generate Observation ##
    B = 100
    from diffusion_policy.model.common.so3_util import random_so3
    obs_r = random_so3(B*3).reshape(B,3,3,3).to(device)
    obs_p = torch.randn(B, 3, 3).to(device)
    obs_f = torch.randn(B, 3, obs_dim).to(device)
    obs ={'obs_r': obs_r, 'obs_p': obs_p, 'obs_f': obs_f}


    act_r = random_so3(B*n_action_steps).reshape(B, n_action_steps, 3, 3).to(device)
    act_p = torch.randn(B, n_action_steps, 3).to(device)
    time = torch.rand(B).to(device)

    ## Run model ##
    model.set_context(obs)
    o1 = model(act_r, act_p, time)

    ##  Rotate scene ##
    from diffusion_policy.model.common.so3_util import random_so3
    rot = random_so3(B).to(device)
    obs_r = torch.einsum('bmn, btnk -> btmk', rot, obs_r)
    act_r = torch.einsum('bmn, btnk -> btmk', rot, act_r)
    obs_p = torch.einsum('bmn, btn -> btm', rot, obs_p)
    act_p = torch.einsum('bmn, btn -> btm', rot, act_p)
    obs ={'obs_r': obs_r, 'obs_p': obs_p, 'obs_f': obs_f}
    model.set_context(obs)
    o2 = model(act_r, act_p, time)

    print(torch.allclose(o1['v'][...,:3],o2['v'][...,:3], atol=1e-5))
    print(torch.allclose(o1['v'][...,3:6],o2['v'][...,3:6], atol=1e-5))







if __name__ == '__main__':
    test()