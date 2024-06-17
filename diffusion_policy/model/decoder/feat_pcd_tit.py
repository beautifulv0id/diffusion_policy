import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.common.se3_util import se3_from_rot_pos
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb

import einops

class FeaturePCloudPolicy(nn.Module):
    def __init__(self, dim=100, depth=1, heads=3, n_obs_steps=1,
                       n_action_steps=1,gripper_out = False,
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
        self.relative_pe_layer = RotaryPositionEncoding3D(dim)

        ## Input Embeddings ##
        self.trajectory_embeddings = nn.Parameter(torch.randn(n_action_steps, dim))
        self.trajectory_encoder = nn.Linear(3+9, dim)
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(dim * n_obs_steps, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )
        self.trajectory_context_head = FFWRelativeCrossAttentionModule(
            dim, heads, num_layers=depth, use_adaln=True
        )
        self.context_self_attn_head = FFWRelativeSelfAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )

        self.gripper_out = gripper_out
        self.ignore_collisions_out = ignore_collisions_out

        out_dim = 6
        if self.gripper_out:
            out_dim += 1
        if self.ignore_collisions_out:
            out_dim += 1

        self.to_out = nn.Linear(dim, out_dim)
        

    def set_context(self, context):
        self.context_pcd = context["context_pcd"]
        self.context_features = context["context_features"]
        self.gripper_pcd = context["gripper_pcd"]
        self.gripper_features = context["gripper_features"]

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
    
    def encode_gripper(self, gripper_features, gripper_pcd, context_features, context_pcd, time_embeddings):
        gripper_features = einops.repeat(
            gripper_features, 'nhist c -> nhist b c', b=gripper_pcd.size(0))
        gripper_pos = self.relative_pe_layer(gripper_pcd)
        context_features = einops.rearrange(
            context_features, 'b n c -> n b c')
        context_pos = self.relative_pe_layer(context_pcd)

        gripper_features = self.gripper_context_head(
            query=gripper_features,
            query_pos=gripper_pos,
            value=context_features,
            value_pos=context_pos)[-1]
        
        gripper_features = einops.rearrange(
            gripper_features, 'nhist b c -> b nhist c'
        )

        gripper_features = gripper_features.flatten(1)

        gripper_features = self.curr_gripper_emb(gripper_features)

        return gripper_features + time_embeddings
    
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
                                context_features, 
                                context_pcd, 
                                gripper_features):
        trajectory_features = einops.rearrange(
            trajectory_features, 'b n c -> n b c')
        trajectory_pos = self.relative_pe_layer(trajectory_pcd)
        context_features = einops.rearrange(
            context_features, 'b n c -> n b c')
        context_pos = self.relative_pe_layer(context_pcd)

        trajectory_features = self.trajectory_context_head(
            query=trajectory_features,
            query_pos=trajectory_pos,
            value=context_features,
            value_pos=context_pos,
            diff_ts=gripper_features)[-1]
        
        trajectory_features = einops.rearrange(
            trajectory_features, 'n b c -> b n c'
        )

        return trajectory_features

    def forward(self, rt, pt, t):
        act_tokens = pt.shape[1]
        time_embeddings = self.time_encoder(t)
        gripper_features = self.gripper_features
        gripper_pcd = self.gripper_pcd
        context_features = self.context_features
        context_pcd = self.context_pcd
        trajectory_pcd = pt
        
        context_features = self.context_self_attn(context_features, context_pcd)

        gripper_features = self.encode_gripper(gripper_features, 
                                                gripper_pcd, 
                                                context_features, 
                                                context_pcd, 
                                                time_embeddings)

        trajectory_features = self.encode_trajectory(rt, pt, time_embeddings)

        trajectory_features = self.trajectory_cross_attn(trajectory_features, 
                                                        trajectory_pcd,
                                                        context_features,
                                                        context_pcd,
                                                        gripper_features)
        
        out = self.to_out(trajectory_features)

        out_dict = {
            "v": out[:, :act_tokens, :6],
            "act_f": trajectory_features[:, :act_tokens, :],
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
    n_obs_steps = 2
    heads = 3
    dim = heads * 20
    model = FeaturePCloudPolicy(dim=dim, heads=heads,n_action_steps=n_action_tokens, n_obs_steps=n_obs_steps)
    from diffusion_policy.model.common.se3_util import random_se3

    B = 120

    Hx = random_se3(B * n_action_tokens).reshape(B, n_action_tokens, 4, 4)
    t = torch.rand(B)
    Xz = torch.randn(B, n_context_tokens, 3)
    Xf = torch.randn(B, n_context_tokens, dim)
    gripper_pcd = torch.randn(B, n_obs_steps, 3)
    gripper_features = torch.randn(n_obs_steps, dim)


    model.set_context({'context_pcd': Xz, 'context_features': Xf,
                          'gripper_pcd': gripper_pcd, 'gripper_features': gripper_features})
    out = model(Hx[..., :3, :3], Hx[..., :3, -1], t)
    v1 = out['v']


    ############### TEST 2 #################
    trans = torch.randn(B, 1, 3)
    Hx2 = Hx
    Hx2[...,:3,-1] = Hx2[...,:3,-1] + trans
    Xz2 = Xz + trans
    gripper_pcd2 = gripper_pcd + trans

    model.set_context({'context_pcd': Xz2, 'context_features': Xf,
                          'gripper_pcd': gripper_pcd2, 'gripper_features': gripper_features})
    out2 = model(Hx2[..., :3, :3], Hx2[..., :3, -1], t)
    v2 = out2['v']

    print(torch.allclose(v1, v2, rtol=1e-05, atol=1e-05))
    print((v1 - v2).abs().max())

if __name__ == '__main__':

    feature_pcloud_git_test()

