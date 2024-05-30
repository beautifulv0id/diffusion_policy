from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import print_dict, dict_apply
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule
import einops
from scipy.spatial import transform


# TODO: proper naming, remove relative? Change to something w/ SE3?
class TransformerLowdimObsRelativeEncoder(ModuleAttrMixin):
    def __init__(self,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
            rotary_embedder: RotaryPositionEncoding3D,
            positional_embedder: SinusoidalPosEmb,
            within_attn : FFWRelativeCrossAttentionModule,
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
        self.re_cross_attn_within = within_attn
        self.re_cross_attn_across = across_attn
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
        return agent_pose, low_dim_pcd
    
    def _encode_grippers(self, gripper, pcd):
        B = gripper.shape[0]
        pcd_pos = self.rotary_embed(pcd)
        pcd_features = self.keypoint_emb(torch.arange(pcd.shape[-2]).to(pcd.device)) 
        pcd_features = einops.repeat(pcd_features, 'n d -> b n d', b=B)
        pcd_features = einops.rearrange(pcd_features, 'b n d -> n b d')
        gripper_pos = self.rotary_embed(gripper[:, :, :3, 3])
        gripper_features = einops.repeat(self.query_emb.weight, "n d -> n b d", b=B)
        gripper_features = self.re_cross_attn_within(query=gripper_features, value=pcd_features, 
                                             query_pos=gripper_pos, value_pos=pcd_pos)[-1]
        return gripper_features
    
    def _encode_current_gripper(self, gripper_features):
        if gripper_features.shape[0] == 1:
            return gripper_features[0]
        
        gripper_pos = self.positional_embedder(
                            torch.arange(self.n_obs_steps, 
                                        dtype=gripper_features.dtype, 
                                        device=gripper_features.device)
                                        ).unsqueeze(1)
        gripper_features = gripper_features + gripper_pos
        gripper_hist = gripper_features[:-1]
        gripper_curr = gripper_features[-1:]
        gripper_features = self.re_cross_attn_across(query=gripper_curr, 
                                            value=gripper_hist)[0]\
                                                .squeeze(0)
        return gripper_features
    
    def forward(self, obs_dict):    
        gripper, low_dim_pcd, = self.process_low_dim_features(obs_dict)
        gripper_features = self._encode_grippers(gripper, low_dim_pcd)
        gripper_features = self._encode_current_gripper(gripper_features)
        return gripper_features

def test():
    bs = 1
    n_obs_steps = 2
    embedding_dim = 60
    n_keypoints = 55
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dict = {
        'agent_pose': torch.rand(bs, n_obs_steps, 4, 4),
        'low_dim_pcd': torch.rand(bs, n_keypoints, 3),
        'low_dim_state': torch.rand(bs, n_obs_steps, 4),
    }
    obs_dict = dict_apply(obs_dict, lambda x: x.to(device))
    obs_encoder = TransformerLowdimObsRelativeEncoder(
        n_obs_steps=n_obs_steps,
        query_embeddings=nn.Embedding(n_obs_steps, embedding_dim),
        keypoint_embeddings=nn.Embedding(n_keypoints, embedding_dim),
        rotary_embedder=RotaryPositionEncoding3D(embedding_dim),
        positional_embedder=SinusoidalPosEmb(embedding_dim),
        within_attn=FFWRelativeCrossAttentionModule(embedding_dim, 3, 2),
        across_attn=FFWRelativeCrossAttentionModule(embedding_dim, 3, 2),
    )
    obs_encoder.to(device)

    out = obs_encoder(obs_dict)

    print_dict(obs_dict)
    print(out.shape)


if __name__ == "__main__":
    test()