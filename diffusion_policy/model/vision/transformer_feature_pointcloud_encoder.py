import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D
from diffusion_policy.model.vision.feature_pyramid_from_rgb import RGB2FeaturePyramid

class TransformerFeaturePointCloudEncoder(ModuleAttrMixin):
    def __init__(self,
                backbone="resnet18",
                embedding_dim=60,
                image_size=(256, 256),
                num_attn_heads=4,
                 ):
        super().__init__()

        self.visual_feature_pyramid = RGB2FeaturePyramid(backbone=backbone, 
                                              image_size=image_size,
                                              embedding_dim=embedding_dim)

        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)


    def interpolate_pcds(self, pcd, size):
        ncam = pcd.shape[1]
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")
        pcd = F.interpolate(pcd, size=size, mode='bilinear')
        pcd = einops.rearrange(pcd, "(bt ncam) c h w -> bt ncam c h w", ncam=ncam)
        return pcd
    
    def _encode_gripper(self, curr_gripper, visual_features, pcds):
        visual_features = einops.rearrange(visual_features, "bt ncam c h w -> bt (ncam h w) c")
        pcds = einops.rearrange(pcds, "bt ncam c h w -> bt (ncam h w) c")
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
            gripper_feats, 'b npt c -> npt b c'
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
    
    def forward(self, obs_dict):
        rgbs = obs_dict['rgbs']
        pcds = obs_dict['pcds']
        curr_gripper = obs_dict['curr_gripper']

        # compute visual features
        visual_features = self.visual_feature_pyramid(rgbs)['res5']

        # interpolate pcds
        pcds = self.interpolate_pcds(pcds, visual_features.shape[-2:])

        # encode gripper
        gripper_features, gripper_pos = self._encode_gripper(curr_gripper, visual_features, pcds)

        return gripper_features, gripper_pos
    
# Example usage
@torch.no_grad()
def test():
    rgb = torch.randn(2, 4, 3, 256, 256)
    pcd = torch.randn(2, 4, 3, 256, 256)
    curr_gripper = torch.randn(2, 1, 4, 4)
    obs_dict = {
        'rgbs': rgb,
        'pcds': pcd,
        'curr_gripper': curr_gripper
    }
    encode = TransformerFeaturePointCloudEncoder()
    
    features, pos = encode(obs_dict)
    
    print("Features shape:")
    print(features.shape)
    print("Position shape:")
    print(pos.shape)

if __name__ == "__main__":
    test()
    print("All tests passed!")