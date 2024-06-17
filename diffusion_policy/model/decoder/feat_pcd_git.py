import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.common.se3_util import se3_from_rot_pos
from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb

class FeaturePCloudPolicy(nn.Module):
    def __init__(self, dim=100, depth=1, heads=3, dim_head=64, mlp_dim=256,
                       context_dim=100, n_action_steps=1,gripper_out = False,
                        ignore_collisions_out = False):

        super().__init__()

        ## Time-Context Embedding ##
        self.time_encoder = SinusoidalPosEmb(dim)

        ## Input Embeddings ##
        self.x_embeddings = nn.Parameter(torch.randn(n_action_steps, dim))

        ## Geometry Invariant Transformer ##
        self.git = GeometryInvariantTransformer(dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 kv_dim=context_dim,
                 return_last_attmap=False)

        self.gripper_out = gripper_out
        self.ignore_collisions_out = ignore_collisions_out

        out_dim = 6
        if self.gripper_out:
            out_dim += 1
        if self.ignore_collisions_out:
            out_dim += 1

        self.to_out = nn.Linear(dim, out_dim)
        

    def set_context(self, context):
        self.context_x = context["obs_p"]
        self.context_features = context["obs_f"]

    def forward(self, rt, pt, t):
        act_tokens = rt.shape[1]
        time_embeddings = self.time_encoder(t)
        x_features = self.x_embeddings[None,...].repeat(rt.shape[0],1,1)
        x_features = x_features + time_embeddings[:,None,:]
        _context_features = self.context_features + time_embeddings[:,None,:]

        x = se3_from_rot_pos(rt, pt)
        
        extras = {'x_poses': x[:,...], 'z_poses': self.context_x,
                  'x_types': 'se3', 'z_types': '3D'}

        features_out = self.git(x_features, z=_context_features, extras=extras)

        out = self.to_out(features_out)

        out_dict = {
            "v": out[:, :act_tokens, :6],
            "act_f": features_out[:, :act_tokens, :],
        }
        if self.gripper_out and self.ignore_collisions_out:
            out_dict["gripper"] = nn.Sigmoid()(out[:, :act_tokens, 6])
            out_dict['ignore_collisions'] = nn.Sigmoid()(out[:, :act_tokens, 7])
        if self.gripper_out:
            out_dict["gripper"] = nn.Sigmoid()(out[:, :act_tokens, 6])
        elif self.ignore_collisions_out:
            out_dict['ignore_collisions'] = nn.Sigmoid()(out[:, :act_tokens, 6])
        
        return out_dict


def feature_pcloud_git_test():
    n_context_tokens = 9
    n_action_tokens = 1
    dim = 100
    model = FeaturePCloudPolicy(dim=dim, n_context=n_context_tokens, n_action_steps=n_action_tokens)
    from diffusion_policy.model.common.se3_util import random_se3

    B = 120

    Hx = random_se3(B * n_action_tokens).reshape(B, n_action_tokens, 4, 4)
    t = torch.rand(B)
    Xz = torch.randn(B, n_context_tokens, 3)
    Xf = torch.randn(B, n_context_tokens, 100)

    model.set_context({'obs_p': Xz, 'obs_f': Xf})
    out = model(Hx[..., :3, :3], Hx[..., :3, -1], t)['v']


    ############### TEST 2 #################
    rot = random_se3(B)
    Hx2 = torch.einsum('bmn,bank->bamk', rot, Hx)
    Xz2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xz) + rot[:, None, :3, -1]

    extras = {'x_poses': Hx2, 'z_poses': Xz2,
              'x_types': 'se3', 'z_types': '3D'}
    model.set_context({'obs_p': Xz2, 'obs_f': Xf})
    out2 = model(Hx2[..., :3, :3], Hx2[..., :3, -1], t)['v']

    print(torch.allclose(out, out2, rtol=1e-05, atol=1e-05))
    print((out - out2).abs().max())


if __name__ == '__main__':

    feature_pcloud_git_test()

