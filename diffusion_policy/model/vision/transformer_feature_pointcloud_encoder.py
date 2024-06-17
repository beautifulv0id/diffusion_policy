import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.vision.feature_pyramid_from_rgb import RGB2FeaturePyramid

class TransformerFeaturePointCloudEncoder(ModuleAttrMixin):
    def __init__(self,
                shape_meta: dict,
                backbone="resnet18",
                feature_layer="res1",
                embedding_dim=60,
                image_size=(256, 256),
                nhist=2,
                num_attn_heads=8,
                gripper_loc_bounds=None,
                 ):
        super().__init__()

        assert ("agent_pose" in shape_meta['obs']), "Must have agent_pos in shape_meta['obs']"
        assert ("rgb" in shape_meta['obs']), "Must have rgb in shape_meta['obs']"
        assert ("pcd" in shape_meta['obs']), "Must have pcd in shape_meta['obs']"

        self.visual_feature_pyramid = RGB2FeaturePyramid(backbone=backbone, 
                                              image_size=image_size,
                                              embedding_dim=embedding_dim)

        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )
        self.goal_gripper_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        self.time_embedder = SinusoidalPosEmb(embedding_dim)

        self.nhist = nhist
        self.shape_meta = shape_meta
        self.feature_layer = feature_layer
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def interpolate_pcds(self, pcd, size):
        ncam = pcd.shape[1]
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")
        pcd = F.interpolate(pcd, size=size, mode='bilinear')
        pcd = einops.rearrange(pcd, "(bt ncam) c h w -> bt ncam c h w", ncam=ncam)
        return pcd
    
    def _encode_gripper(self, curr_gripper, visual_features, pcds):
        gripper_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(curr_gripper), 1, 1
        )
        gripper_pos = curr_gripper[...,:3,-1]
        return self.encode_gripper(gripper_feats=gripper_feats, gripper_pos=gripper_pos, context_feats=visual_features, context_pos=pcds)
    
    def encode_gripper(self, gripper_feats, gripper_pos, context_feats, context_pos):
        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper_pos)
        context_pos = self.relative_pe_layer(context_pos)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b nhist c -> nhist b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )
        return gripper_feats, gripper_pos
    

    def _encode_current_gripper(self, gripper_feats):
        goal_feats = self.goal_gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper_feats), 1, 1
        ).to(gripper_feats.device)
        return self.encode_current_gripper(goal_feats, gripper_feats)
    
    def encode_current_gripper(self, current_feats, context_feats):
        time_pos = self.time_embedder(torch.arange(self.nhist, device=current_feats.device)).unsqueeze(0)
        context_feats = context_feats + time_pos

        current_feats = einops.rearrange(
            current_feats, 'b 1 c -> 1 b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )

        current_feats = self.goal_gripper_head(
            query=current_feats, value=context_feats,
        )[-1]
        current_feats = current_feats.squeeze(0)
        return current_feats
    
    def encode_rgbs(self, rgbs):
        ncam = rgbs.shape[1]
        rgbs = einops.rearrange(rgbs, "bt ncam c h w -> (bt ncam) c h w")
        visual_features = self.visual_feature_pyramid(rgbs)[self.feature_layer]
        visual_features = einops.rearrange(visual_features, "(bt ncam) c h w -> bt ncam c h w", ncam=ncam)
        return visual_features
    
    
    def forward(self, obs_dict):
        rgbs = obs_dict['rgb']
        pcds = obs_dict['pcd']
        curr_gripper = obs_dict['agent_pose']

        # compute visual features
        visual_features = self.encode_rgbs(rgbs)

        # interpolate pcds
        pcds = self.interpolate_pcds(pcds, visual_features.shape[-2:])

        visual_features = einops.rearrange(visual_features, "bt ncam c h w -> bt (ncam h w) c")
        pcds = einops.rearrange(pcds, "bt ncam c h w -> bt (ncam h w) c")

        # encode gripper
        gripper_features, gripper_pos = self._encode_gripper(curr_gripper, visual_features, pcds)

        # encode current gripper
        gripper_features = self._encode_current_gripper(gripper_features)

        return gripper_features
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.nhist
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,To) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

# Example usage
@torch.no_grad()
def test():
    nhist = 4
    ncam = 4
    h, w = 128, 128
    rgb = torch.randn(2, ncam, 3, h, w)
    pcd = torch.randn(2, ncam, 3, h, w)
    curr_gripper = torch.randn(2, nhist, 4, 4)
    obs_dict = {
        'rgb': rgb,
        'pcd': pcd,
        'agent_pose': curr_gripper
    }
    shape_meta = {
        'obs': {
            'rgb': {'shape': (3, h, w)},
            'pcd': {'shape': (3, h, w)},
            'agent_pose': {'shape': (4, 4)}
        }
    }
    encoder = TransformerFeaturePointCloudEncoder(
        shape_meta=shape_meta,
        nhist=nhist, 
        embedding_dim=192,
        image_size=(h, w),
        gripper_loc_bounds=[[
            -1,
            -1,
            -1
        ],
        [
            1,
            1,
            1
        ]])

    visual_features = encoder.encode_rgbs(rgb)
    pcd = encoder.interpolate_pcds(pcd, visual_features.shape[-2:])   

    visual_features = einops.rearrange(visual_features, "bt ncam c h w -> bt (ncam h w) c")
    pcd = einops.rearrange(pcd, "bt ncam c h w -> bt (ncam h w) c")

    features, pos = encoder._encode_gripper(curr_gripper, visual_features, pcd)
    features = encoder._encode_current_gripper(features)

    print("Features shape:")
    print(features.shape)
    print("Position shape:")
    print(pos.shape)

    features = encoder(obs_dict)
    print("Features shape:")
    print(features.shape)


if __name__ == "__main__":
    import numpy as np

    # a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # m = torch.tensor([[True, False, False], [False, True, True], [True, False, True]])
    # print(mask_and_fill_remaining(a, m))

    # a = torch.randn(2, 4, 3)
    # m = a > 0
    # print(m)
    # print(fill_mask(m))
    # print(mask_and_fill_remaining(a, m))
    test()
    print("All tests passed!")