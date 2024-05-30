from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, print_dict
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.common.layers import FFWRelativeSelfAttentionModule, FFWRelativeCrossAttentionModule
import einops
from scipy.spatial import transform

# TODO: proper naming, remove relative? Change to something w/ SE3?
class TransformerSelfAttnLowdimEncoder(ModuleAttrMixin):
    def __init__(self,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
            rotary_embedder: RotaryPositionEncoding3D,
            positional_embedder: SinusoidalPosEmb,
            within_attn : FFWRelativeSelfAttentionModule,
            across_attn : FFWRelativeCrossAttentionModule,
        ):
        """
        Assumes rgb input: B,To,C,H,W
        Assumes low_dim input: B,To,D
        """
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.rotary_embedder = rotary_embedder
        self.query_emb = query_embeddings
        self.keypoint_emb = keypoint_embeddings
        self.within_attn = within_attn
        self.across_attn = across_attn
        self.positional_embedder = positional_embedder

    def rotary_embed(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, Da)
        Returns:
            torch.Tensor: (B, N, D, 2)
        """
        return self.rotary_embedder(x)
    
    
    def process_low_dim_features(self, obs_dict):
        agent_pose = obs_dict['agent_pose']
        low_dim_pcd = obs_dict['low_dim_pcd']
        keypoint_features = self.keypoint_emb(torch.arange(low_dim_pcd.shape[-2]).to(low_dim_pcd.device))
        keypoint_features = einops.repeat(keypoint_features, 'n d -> b n d', b=agent_pose.shape[0]).to(agent_pose.device)
        low_dim_features = obs_dict['low_dim_state']
        return agent_pose, low_dim_pcd, keypoint_features, low_dim_features
    
    def forward(self, obs_dict):    
        gripper, low_dim_pcd, keypoint_features, _ = \
            self.process_low_dim_features(obs_dict)
        gripper = torch.cat([gripper[:,-1:], gripper[:,:-1]], dim=1)
        B = gripper.shape[0]
        gripper_features = einops.repeat(self.query_emb.weight, "n d -> b n d", b=B).to(self.device)
        query = torch.cat([gripper_features, keypoint_features], dim=1)
        gripper_pos = self.rotary_embed(gripper[:, :, :3, 3])
        low_dim_pcd_pos = self.rotary_embed(low_dim_pcd)
        query_pos = torch.cat([gripper_pos, low_dim_pcd_pos], dim=1)

        query = einops.rearrange(query, 'b n d -> n b d')
        features = self.within_attn(query=query, query_pos=query_pos)[-1]

        query = features[:1]
        value = features[1:]
        query_pos, value_pos = query_pos[:, :1], query_pos[:, 1:]
        features = self.across_attn(query=query, value=value, query_pos=query_pos, value_pos=value_pos)[-1]
        features = einops.rearrange(features, '1 b d -> b d')
        return features
        
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.n_obs_steps
        agent_pose_shape = tuple(obs_shape_meta.pop('agent_pose')['shape'])
        agent_pose = torch.zeros(
            (batch_size,To) + agent_pose_shape, 
            dtype=self.dtype,
            device=self.device)
        example_obs_dict['agent_pose'] = agent_pose
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

def test():    
    bs = 1
    n_obs_steps = 2
    embedding_dim = 60
    n_keypoints = 55
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dict = {
        'agent_pose': torch.rand(bs, n_obs_steps, 4, 4),
        'low_dim_pcd': torch.rand(bs, n_keypoints, 3),
        'low_dim_state': torch.rand(bs, n_obs_steps, 3),
    }
    obs_dict = dict_apply(obs_dict, lambda x: x.to(device))

    encoder = TransformerSelfAttnLowdimEncoder(
        n_obs_steps=n_obs_steps,
        query_embeddings=nn.Embedding(n_obs_steps, embedding_dim),
        keypoint_embeddings=nn.Embedding(n_keypoints, embedding_dim),
        rotary_embedder=RotaryPositionEncoding3D(embedding_dim),
        positional_embedder=SinusoidalPosEmb(embedding_dim),
        within_attn=FFWRelativeSelfAttentionModule(
            embedding_dim=embedding_dim,
            num_attn_heads=3,
            num_layers=1,),
        across_attn=FFWRelativeCrossAttentionModule(
            embedding_dim=embedding_dim,
            num_attn_heads=3,
            num_layers=1,),
    )
    encoder.to(device)
    out = encoder.forward(obs_dict)
    print_dict(obs_dict)
    print(out.shape)


if __name__ == "__main__":
    test()