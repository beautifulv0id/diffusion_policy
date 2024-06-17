import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

import einops

from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.common.se3_util import se3_from_rot_pos
from diffusion_policy.model.common.layers import FFWRelativeSelfAttentionModule
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb


class FeaturePCloudPolicy(nn.Module):
    def __init__(self, dim=100, depth=1, heads=3, dim_head=64, mlp_dim=256,
                       n_action_steps=1, n_obs_steps=1, gripper_out = False,
                        ignore_collisions_out = False):

        super().__init__()
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        ## Time-Context Embedding ##
        self.time_encoder = SinusoidalPosEmb(dim)

        ## Input Embeddings ##
        self.x_embeddings = nn.Parameter(torch.randn(n_action_steps, dim))
        self.trajectory_embeddings = nn.Parameter(torch.randn(n_action_steps, dim))
        self.traj_time_emb = SinusoidalPosEmb(dim)
        self.trajectory_encoder = nn.Linear(16, dim)

        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(dim * n_obs_steps, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        ## Geometry Invariant Transformer ##
        self.gripper_context_head = GeometryInvariantTransformer(dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 kv_dim=dim,
                 return_last_attmap=False)
        
        self.trajectory_context_head = GeometryInvariantTransformer(dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 kv_dim=dim,
                 use_adaln=True,
                 return_last_attmap=False)

        self.context_self_attn_head = FFWRelativeSelfAttentionModule(
            dim, heads, num_layers=depth, use_adaln=False
        )
        self.relative_pe_layer = RotaryPositionEncoding3D(dim)

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
        self.gripper_rot = context["gripper_rot"]
        self.gripper_features = context["gripper_features"]

    def encode_gripper(self, gripper_features, gripper_pose, context_features, context_pcd, time_embeddings):
        extras = {'x_poses': gripper_pose, 'z_poses': context_pcd,
                  'x_types': 'se3', 'z_types': '3D'}
        gripper_features = einops.repeat(
            gripper_features, 'nhist c -> b nhist c', b=gripper_pose.size(0))

        gripper_features = self.gripper_context_head(
            x=gripper_features,
            z=context_features,
            extras=extras
        )
        
        gripper_features = gripper_features.flatten(1)

        gripper_features = self.curr_gripper_emb(gripper_features)

        return gripper_features + time_embeddings

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
    
    def encode_trajectory(self, trajectory, time_embeddings):
        batch_size = time_embeddings.shape[0]
        trajectory_features = self.trajectory_encoder(trajectory.flatten(1))
        trajectory_embeddings = self.trajectory_embeddings[None,...].repeat(batch_size,1,1)
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, self.n_action_steps, device=time_embeddings.device)
        )[None].repeat(batch_size, 1, 1)
        return traj_time_pos + trajectory_embeddings + time_embeddings[:,None,:]

    def trajectory_cross_attn(self, 
                                trajectory_features, 
                                trajectory_pose, 
                                context_features, 
                                context_pcd, 
                                gripper_features):
        extras = {'x_poses': trajectory_pose, 'z_poses': context_pcd,
                    'x_types': 'se3', 'z_types': '3D'}
        trajectory_features = self.trajectory_context_head(trajectory_features, z=context_features, extras=extras, diff_ts=gripper_features)

        return trajectory_features


    def forward(self, rt, pt, t):
        act_tokens = pt.shape[1]
        time_embeddings = self.time_encoder(t)
        gripper_features = self.gripper_features
        gripper_pose = se3_from_rot_pos(self.gripper_rot, self.gripper_pcd)
        context_features = self.context_features
        context_pcd = self.context_pcd
        trajectory_pose = se3_from_rot_pos(rt, pt)

        context_features = self.context_self_attn(context_features, context_pcd)

        gripper_features = self.encode_gripper(gripper_features, gripper_pose, context_features, context_pcd, time_embeddings)

        trajectory_embedding = self.encode_trajectory(trajectory_pose, time_embeddings)

        trajectory_features = self.trajectory_cross_attn(
            trajectory_embedding, trajectory_pose, context_features, context_pcd, gripper_features)
        
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


def feature_pcloud_git_test():
    n_context_tokens = 9
    n_action_tokens = 1
    n_obs_steps = 2
    heads = 3
    dim = heads * 20
    model = FeaturePCloudPolicy(dim=dim, heads=heads,n_action_steps=n_action_tokens, n_obs_steps=n_obs_steps)
    from diffusion_policy.model.common.se3_util import random_se3
    from diffusion_policy.model.common.so3_util import random_so3

    B = 120

    Hx = random_se3(B * n_action_tokens).reshape(B, n_action_tokens, 4, 4)
    t = torch.rand(B)
    Xz = torch.randn(B, n_context_tokens, 3)
    Xf = torch.randn(B, n_context_tokens, dim)
    gripper_pcd = torch.randn(B, n_obs_steps, 3)
    gripper_rot = random_so3(B * n_obs_steps).reshape(B, n_obs_steps, 3, 3)
    gripper_features = torch.randn(n_obs_steps, dim)

    model.set_context({'context_pcd': Xz, 'context_features': Xf,
                          'gripper_pcd': gripper_pcd, 'gripper_rot': gripper_rot,
                            'gripper_features': gripper_features})
    out = model(Hx[..., :3, :3], Hx[..., :3, -1], t)
    v1 = out['v']


    ############### TEST 2 #################
    rot = random_se3(B)
    # trans = torch.randn(B, 1, 3)
    # Hx2 = Hx
    # Hx2[...,:3,-1] = Hx2[...,:3,-1] + trans
    # Xz2 = Xz + trans
    # gripper_pcd2 = gripper_pcd + trans
    # gripper_rot2 = gripper_rot

    Hx2 = torch.einsum('bmn,bank->bamk', rot, Hx)
    Xz2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xz) + rot[:, None, :3, -1]
    gripper_pcd2 = torch.einsum('bmn,bhn->bhm', rot[..., :3, :3], gripper_pcd) + rot[:, None, :3, -1]
    gripper_rot2 = torch.einsum('bmn,bank->bamk', rot[..., :3, :3], gripper_rot)

    model.set_context({'context_pcd': Xz2, 'context_features': Xf,
                          'gripper_pcd': gripper_pcd2, 'gripper_rot': gripper_rot2,
                            'gripper_features': gripper_features})
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

    feature_pcloud_git_test()

