import einops
from torch import nn
import torch
from torch.nn import functional as F
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.model.vision.resnet_wrapper import load_resnet50, load_resnet18
from diffusion_policy.model.vision.clip_wrapper import load_clip, CLIP_RES_TO_DIM


class FeaturePCDEncoder(ModuleAttrMixin):
    
    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 feature_res="res2"):
        super().__init__()

        assert feature_res in ["res1", "res2"]

        # 3D relative positional embeddings
        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feature_res = feature_res
        self.out_dim = CLIP_RES_TO_DIM[self.feature_res]
        self.obs_features = nn.Parameter(torch.randn(1, self.out_dim))
        
    def forward(self, rgb, pcd):
        if rgb is not None:
            return self.get_feature_pcd(rgb, pcd)
        else:
            return self.get_lowdim_feature_pcd(pcd)
    
    def get_lowdim_feature_pcd(self, pcd):
        batch, npts = pcd.shape[0:2]
        feats = self.obs_features.unsqueeze(0).expand(batch, npts, -1)
        return feats, pcd


    def get_feature_pcd(self, rgb, pcd):
        """
        Compute visual features

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats: (B, ncam, F, H, W)
            - pcd: (B, ncam * H * W, 3), resampled point cloud
        """            
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)
        rgb_features = rgb_features[self.feature_res]

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        # Interpolate xy-depth to get the locations for this level
        feat_h, feat_w = rgb_features.shape[-2:]
        pcd = F.interpolate(
            pcd,
            (feat_h, feat_w),
            mode='bilinear'
        )

        # Merge different cameras for clouds, separate for rgb features
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        rgb_features = einops.rearrange(
            rgb_features,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )

        return rgb_features, pcd
    