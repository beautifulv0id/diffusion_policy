from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import print_dict
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.common.layers import FFWRelativeSelfAttentionModule, FFWRelativeCrossAttentionModule
from einops import rearrange, repeat
from scipy.spatial import transform

class LowDimPoseEmbedder(ModuleAttrMixin):
    def __init__(self,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
            **kwargs
        ):
        """
        Assumes rgb input: B,To,C,H,W
        Assumes low_dim input: B,To,D
        """
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.query_emb = query_embeddings
        self.keypoint_emb = keypoint_embeddings
    
    def forward(self, obs_dict):    
        gripper = obs_dict['agent_pose']
        keypoint_pose = obs_dict['keypoint_poses']
        B = gripper.shape[0]
        keypoint_features = repeat(self.keypoint_emb.weight, "n d -> b n d", b=B).to(self.device)
        gripper_features = repeat(self.query_emb.weight, "n d -> b n d", b=B).to(self.device)
        features = torch.cat([gripper_features, keypoint_features], dim=1)
        positions = torch.cat([gripper[:,:,:3,3], keypoint_pose[:,:,:3,3]], dim=1)
        rotations = torch.cat([gripper[:,:,:3,:3], keypoint_pose[:,:,:3,:3]], dim=1)
        ret = {         
            'obs_f': features,
            'obs_p': positions,
            'obs_r': rotations,
        }
        return ret

def test():    
    n_action_steps = 1
    n_obs_steps = 1
    embedding_dim = 60
    n_keypoints = 3
    B = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = LowDimPoseEmbedder(
        n_obs_steps=n_obs_steps,
        query_embeddings=nn.Embedding(n_obs_steps, embedding_dim),
        keypoint_embeddings=nn.Embedding(n_keypoints, embedding_dim),
    )
    encoder = encoder.to(device)
    example_obs_dict = {
        "agent_pose": torch.randn(B, 1, 4, 4).to(device),
        "keypoint_poses": torch.randn(B, 3, 4, 4).to(device),
    }
    obs = encoder.forward(example_obs_dict)

    print_dict(obs)

    # from diffusion_policy.model.flow_matching.flow_matching_git import FlowMatchingInvariantPointTransformer
    # from diffusion_policy.model.common.so3_util import random_so3
    # n_action_steps = 1

    # act_r = random_so3(B*n_action_steps).reshape(B, n_action_steps, 3, 3).to(device)
    # act_p = torch.randn(B, n_action_steps, 3).to(device)
    # time = torch.rand(B).to(device)

    # ## Run model ##
    # ipt = FlowMatchingInvariantPointTransformer(obs_dim=60, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, gripper_out=True,ignore_collision_out=True).to(device)
    # ipt.set_context(obs)
    # out = ipt(act_r, act_p, time)
    # print_dict(out)

if __name__ == "__main__":
    test()