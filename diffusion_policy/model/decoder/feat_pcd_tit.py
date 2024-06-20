import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.common.se3_util import se3_from_rot_pos
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention
import einops

class FeaturePCloudPolicy(nn.Module):
    def __init__(self, dim=100, depth=1, heads=3, n_obs_steps=1,
                       horizon=1,gripper_out = False,
                        ignore_collisions_out = False):

        super().__init__()

        ## Time-Context Embedding ##
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(dim)
        self.gripper_time_emb = SinusoidalPosEmb(dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(dim)

        ## Input Embeddings ##
        self.trajectory_embeddings = nn.Parameter(torch.randn(horizon, dim))
        self.trajectory_encoder = nn.Linear(3+9, dim)
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.low_dim_state_emb = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )
        self.trajectory_context_head = FFWRelativeCrossAttentionModule(
            dim, heads, num_layers=depth, use_adaln=True
        )
        self.trajectory_self_attn_head = FFWRelativeSelfAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )
        self.context_self_attn_head = FFWRelativeSelfAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )

        self.trajectory_inv_point_self_attn_head = InvariantPointTransformer(dim, depth, heads, heads, kv_dim=dim,
                                    dropout=0.1, attention_module=InvariantPointAttention, point_dim=3)

        self.trajectory_gripper_self_attn_head = InvariantPointTransformer(dim, depth, heads, heads, kv_dim=dim,
                                    dropout=0.1, attention_module=InvariantPointAttention, point_dim=3)
        
        self.gripper_out = gripper_out
        self.ignore_collisions_out = ignore_collisions_out

        out_dim = 6
        if self.gripper_out:
            out_dim += 1
        if self.ignore_collisions_out:
            out_dim += 1

        self.to_out = nn.Linear(dim, out_dim)
        
        self._keys_to_ignore_on_save = ['context_features', 'gripper_features']

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in self._keys_to_ignore_on_save:
            state_dict.pop(prefix + key, None)
        return state_dict
    
    def set_context(self, context):
        self.context_pcd = context["context_pcd"]
        self.context_features = context["context_features"]
        self.gripper_pcd = context["gripper_pcd"]
        self.gripper_rot = context["gripper_rot"]
        self.gripper_features = context["gripper_features"]
        self.low_dim_state = context["low_dim_state"]

    def context_self_attn(self, context_features, context_pcd):
        context_features = einops.rearrange(
            context_features, 'b n c -> n b c')
        context_pos = self.relative_pe_layer(context_pcd)

        context_features = self.context_self_attn_head(
            query=context_features,
            query_pos=context_pos)[-1]
        
        context_features = einops.rearrange(
            context_features, 'n b c -> b n c'
        )
        return context_features
    
    def encode_gripper(self, gripper_features, gripper_pcd, context_features, context_pcd, low_dim_state, time_embeddings):
        gripper_features = einops.rearrange(
            gripper_features, 'b nhist c -> nhist b c')
        gripper_pos = self.relative_pe_layer(gripper_pcd)
        context_features = einops.rearrange(
            context_features, 'b n c -> n b c')
        context_pos = self.relative_pe_layer(context_pcd)

        low_dim_state_emb = self.low_dim_state_emb(low_dim_state)
        low_dim_state_emb = einops.rearrange(
            low_dim_state_emb, 'b nhist c -> nhist b c')

        gripper_features = gripper_features + low_dim_state_emb

        gripper_features = self.gripper_context_head(
            query=gripper_features,
            query_pos=gripper_pos,
            value=context_features,
            value_pos=context_pos)[-1]
        
        gripper_features = einops.rearrange(
            gripper_features, 'nhist b c -> b nhist c'
        )

        gripper_features = self.curr_gripper_emb(gripper_features)

        gripper_time_pos = self.gripper_time_emb(
            torch.arange(0, gripper_features.size(1), device=gripper_features.device)
        )[None].repeat(len(gripper_features), 1, 1)

        return gripper_features + time_embeddings[:,None,:] + gripper_time_pos
    
    def encode_trajectory(self, rot, pos, time_embeddings):
        rot = torch.flatten(rot, -2)
        trajectory = torch.cat([rot, pos], -1)  
        trajectory_features = self.trajectory_encoder(trajectory)
        trajectory_embeddings = self.trajectory_embeddings[None,...].repeat(trajectory_features.shape[0],1,1)
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, trajectory_features.size(1), device=trajectory_features.device)
        )[None].repeat(len(trajectory_features), 1, 1)
        return trajectory_features + traj_time_pos + trajectory_embeddings + time_embeddings[:,None,:]
    
    def trajectory_cross_attn(self, 
                                trajectory_features, 
                                trajectory_pcd, 
                                pcd_features, 
                                pcd, 
                                gripper_features,
                                gripper_pcd):
        trajectory_features = einops.rearrange(
            trajectory_features, 'b n c -> n b c')
        trajectory_pos = self.relative_pe_layer(trajectory_pcd)

        context_features = torch.cat([pcd_features, gripper_features], dim=1)
        context_pos = torch.cat([pcd, gripper_pcd], dim=1)

        context_features = einops.rearrange(
            context_features, 'b n c -> n b c')
        context_pos = self.relative_pe_layer(context_pos)

        trajectory_features = self.trajectory_context_head(
            query=trajectory_features,
            query_pos=trajectory_pos,
            value=context_features,
            value_pos=context_pos)[-1]
        
        trajectory_features = einops.rearrange(
            trajectory_features, 'n b c -> b n c'
        )

        return trajectory_features
    
    def trajectory_self_attn(self, trajectory_features, trajectory_pcd):

        trajectory_features = einops.rearrange(
            trajectory_features, 'b n c -> n b c')
        trajectory_pos = self.relative_pe_layer(trajectory_pcd)

        attn_mask = torch.full((trajectory_features.size(0), trajectory_features.size(0)), -np.inf, device=trajectory_features.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        trajectory_features = self.trajectory_self_attn_head(
            query=trajectory_features,
            query_pos=trajectory_pos,
            attn_mask=attn_mask)[-1]
        
        trajectory_features = einops.rearrange(
            trajectory_features, 'n b c -> b n c'
        )

        return trajectory_features
    
    def trajectory_inv_point_self_attn(self, trajectory_features, trajectory_pcd, trajectory_rot):
        trajectory_pose = {
            'rotations': trajectory_rot,
            'translations': trajectory_pcd
        }

        attn_mask = torch.ones((trajectory_features.size(1), trajectory_features.size(1)), device=trajectory_features.device)
        attn_mask = torch.tril(attn_mask).bool()

        trajectory_features = self.trajectory_inv_point_self_attn_head(
            trajectory_features,
            trajectory_pose,
            mask=attn_mask)
        
        return trajectory_features

    def trajectory_gripper_self_attn(self, trajectory_features, trajectory_rot, trajectory_pcd, gripper_features, gripper_rot, gripper_pcd):
        poses = {
            'rotations': torch.cat([trajectory_rot, gripper_rot], dim=1),
            'translations': torch.cat([trajectory_pcd, gripper_pcd], dim=1)
        }

        features = torch.cat([trajectory_features, gripper_features], dim=1)

        Nt = trajectory_features.size(1)
        Ng = self.gripper_features.size(0)

        attn_mask = torch.ones((Nt + Ng, Nt + Ng), device=trajectory_features.device)
        attn_mask[:Nt, :Nt] = torch.tril(attn_mask[:Nt, :Nt])
        attn_mask = attn_mask.bool()

        trajectory_features = self.trajectory_gripper_self_attn_head(
            features,
            poses,
            mask=attn_mask)
        
        return trajectory_features

    def forward(self, rt, pt, t):
        act_tokens = pt.shape[1]
        time_embeddings = self.time_encoder(t)
        gripper_features = self.gripper_features
        gripper_pcd = self.gripper_pcd
        gripper_rot = self.gripper_rot
        context_features = self.context_features
        context_pcd = self.context_pcd
        low_dim_state = self.low_dim_state
        trajectory_pcd = pt
        trajectory_rot = rt

        gripper_features = einops.repeat(
            gripper_features, 'nhist c -> b nhist c', b=gripper_pcd.size(0))
        context_features = einops.repeat(
            context_features, 'n c -> b n c', b=gripper_pcd.size(0))
        
        context_features = self.context_self_attn(context_features, context_pcd)

        gripper_features = self.encode_gripper(gripper_features, 
                                                gripper_pcd, 
                                                context_features, 
                                                context_pcd, 
                                                low_dim_state,
                                                time_embeddings)

        trajectory_embedding = self.encode_trajectory(rt, pt, time_embeddings)

        trajectory_features = self.trajectory_cross_attn(trajectory_embedding, 
                                                        trajectory_pcd,
                                                        context_features,
                                                        context_pcd,
                                                        gripper_features,
                                                        gripper_pcd)
        
        # trajectory_features = self.trajectory_self_attn(trajectory_features, trajectory_pcd)

        trajectory_features = self.trajectory_gripper_self_attn(trajectory_features, 
                                                                trajectory_rot,
                                                                trajectory_pcd,
                                                                gripper_features,
                                                                gripper_rot,
                                                                gripper_pcd)
        
        out = self.to_out(trajectory_features)

        out_dict = {
            "v": out[:, :act_tokens, :6],
            "act_f": trajectory_features[:, :act_tokens, :],
            "context_features": context_features,
            "gripper_features": gripper_features,
            "trajectory_embedding": trajectory_embedding
        }
        if self.gripper_out and self.ignore_collisions_out:
            out_dict["gripper"] = nn.Sigmoid()(out[:, :act_tokens, 6])
            out_dict['ignore_collisions'] = nn.Sigmoid()(out[:, :act_tokens, 7])
        if self.gripper_out:
            out_dict["gripper"] = nn.Sigmoid()(out[:, :act_tokens, 6])
        elif self.ignore_collisions_out:
            out_dict['ignore_collisions'] = nn.Sigmoid()(out[:, :act_tokens, 6])
        
        return out_dict


def feature_pcloud_tit_test():
    n_context_tokens = 9
    horizon = 16
    n_obs_steps = 2
    heads = 3
    dim = heads * 20
    model = FeaturePCloudPolicy(dim=dim, heads=heads,horizon=horizon, n_obs_steps=n_obs_steps)
    from diffusion_policy.model.common.se3_util import random_se3
    from diffusion_policy.model.common.so3_util import random_so3

    B = 120

    Hx = random_se3(B * horizon).reshape(B, horizon, 4, 4)
    t = torch.rand(B)
    Xz = torch.randn(B, n_context_tokens, 3)
    Xf = torch.randn(n_context_tokens, dim)
    gripper_pcd = torch.randn(B, n_obs_steps, 3)
    gripper_rot = random_so3(B * n_obs_steps).reshape(B, n_obs_steps, 3, 3)
    gripper_features = torch.randn(n_obs_steps, dim)
    low_dim_state = torch.randn(B, n_obs_steps, 3)


    model.set_context({'context_pcd': Xz, 'context_features': Xf, 'gripper_rot' : gripper_rot,
                          'gripper_pcd': gripper_pcd, 'gripper_features': gripper_features,
                          'low_dim_state': low_dim_state})
    out = model(Hx[..., :3, :3], Hx[..., :3, -1], t)
    v1 = out['v']


    ############### TEST 2 #################
    trans = torch.randn(B, 1, 3)
    Hx2 = Hx
    Hx2[...,:3,-1] = Hx2[...,:3,-1] + trans
    Xz2 = Xz + trans
    gripper_pcd2 = gripper_pcd + trans

    model.set_context({'context_pcd': Xz2, 'context_features': Xf, 'gripper_rot' : gripper_rot,
                          'gripper_pcd': gripper_pcd2, 'gripper_features': gripper_features,
                            'low_dim_state': low_dim_state})
    out2 = model(Hx2[..., :3, :3], Hx2[..., :3, -1], t)
    v2 = out2['v']

    print(torch.allclose(out['context_features'], out2['context_features']))
    print((out['context_features'] - out2['context_features']).abs().max())
    print(torch.allclose(out['gripper_features'], out2['gripper_features']))
    print((out['gripper_features'] - out2['gripper_features']).abs().max())
    print(torch.allclose(out['trajectory_embedding'], out2['trajectory_embedding']))
    print((out['trajectory_embedding'] - out2['trajectory_embedding']).abs().max())

    print(torch.allclose(v1, v2, rtol=1e-05, atol=1e-05))
    print((v1 - v2).abs().max())

if __name__ == '__main__':

    Nt = 5
    Ng = 2
    attn_mask = torch.ones((Nt + Ng, Nt + Ng))
    attn_mask[:Nt, :Nt] = torch.tril(attn_mask[:Nt, :Nt]).bool()

    print(attn_mask)

    feature_pcloud_tit_test()

