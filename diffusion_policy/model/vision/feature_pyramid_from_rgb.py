# Code from https://github.com/zhouxian/act3d-chained-diffuser

# import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from diffusion_policy.model.vision.resnet import load_resnet50, load_resnet18
# from diffusion_policy.model.vision.clip import load_clip


class RGB2FeaturePyramid(nn.Module):
    def __init__(self,
                 backbone="resnet18",
                 image_size=(256, 256),
                 embedding_dim=60,
                 ):
        super().__init__()
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [(128, 128), (256, 256)]

        self.image_size = image_size

        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]


    def forward(self, rgb):
        ncam = rgb.shape[1]
        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        for k,v in rgb_features.items():
            rgb_features[k] = einops.rearrange(v, "(bt ncam) c h w -> bt ncam c h w", ncam=ncam)
        return rgb_features
    
# Example usage
@torch.no_grad()
def test():
    def print_dict(x, indent=0):
        for k in x.keys():
            if isinstance(x[k], dict):
                print(" "*3*indent+k+":")
                print_dict(x[k], indent+1)
            else:
                print(" "*3*indent+k+":", x[k].numpy().shape)

    
    rgb = torch.randn(2, 4, 3, 256, 256)
    encoder = RGB2FeaturePyramid()
    features = encoder(rgb)
    
    print("Feature pyramid:")
    print_dict(features)

if __name__ == "__main__":
    test()
    print("All tests passed!")