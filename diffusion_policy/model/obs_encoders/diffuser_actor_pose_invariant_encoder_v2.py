import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention
from diffusion_policy.model.vision.resnet import load_resnet50, load_resnet18
from diffusion_policy.model.vision.clip_wrapper import load_clip

class DiffuserActorEncoder(ModuleAttrMixin):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_sampling_level=3,
                 nhist=3,
                 num_attn_heads=8,
                 fps_subsampling_factor=5,
                 quaternion_format='xyzw'):
        super().__init__()

        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor

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
            self.feature_map_pyramid = self.coarse_feature_map
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        head_dim = embedding_dim // num_attn_heads
        assert head_dim * num_attn_heads == embedding_dim, "embed_dim must be divisible by num_heads"

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = InvariantPointTransformer(
            embedding_dim, 3, num_attn_heads, head_dim, kv_dim=embedding_dim, use_adaln=False,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3)

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        self._quaternion_format = quaternion_format
        
    def forward(self):
        return None

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 4, 4)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )


        poses_x = {'rotations': gripper[..., :3, :3], 
                   'translations': gripper[..., :3, 3], 'types': 'se3'}
        poses_z = {'rotations': torch.eye(3).reshape(1,1,3,3).repeat(context.size(0), context.size(1), 1, 1).to(context.device),
                   'translations': context, 'types': 'se3'}
        point_mask_z = torch.ones_like(context[..., 0]).bool()

        gripper_feats = self.gripper_context_head(x=gripper_feats, poses_x=poses_x, z=context_feats, poses_z=poses_z, point_mask_z=point_mask_z)

        return gripper_feats


    def mask_out_features_pcd(self, mask, rgb_features, pcd, n_min=0, n_max=1000000):
        """
        Masks out features and point cloud data based on a given mask.

        Args:
            mask (torch.Tensor): (B, ncam, 1, H, W)
            rgb_features (torch.Tensor): (B, ncam, F, H, W)
            pcd (torch.Tensor): (B, ncam, 3, H, W)
            n_min (int, optional): 
            n_max (int, optional): 
        Returns:
            rgb_features (torch.Tensor): (B, N, F)
            pcd (torch.Tensor): (B, N, 3)
        """
        this_mask = mask.clone()
        b, v, _, h, w = rgb_features.shape
        rgb_features = einops.rearrange(rgb_features, 'b v c h w -> b (v h w) c')
        this_mask = F.interpolate(this_mask.flatten(0, 1).float(), (h, w), mode='nearest').bool().reshape(b, v, h, w)

        B = this_mask.size(0)
        n = this_mask.view(B, -1).count_nonzero(dim=-1)
        n_sample = torch.clamp(n.max(), n_min, n_max)
        diff = n_sample - n
        neg_inds = (~this_mask.view(B, -1)).nonzero(as_tuple=True)[1]
        neg_indsn = (~this_mask.view(B, -1)).count_nonzero(dim=-1)
        neg_indsn = torch.cat([torch.zeros(1, device=mask.device), torch.cumsum(neg_indsn, dim=0)])
        idx0 = torch.tensor([], device=mask.device, dtype=torch.int)
        idx1 = torch.tensor([], device=mask.device, dtype=torch.int)
        for i in range(B):
            offset = diff[i].int().item()
            if offset > 0:
                neg_i = neg_indsn[i].int().item()
                idx0 = torch.cat((idx0, torch.full((offset,), i, device=mask.device)))
                idx1 = torch.cat((idx1, neg_inds[neg_i:neg_i + offset]))
        fill_inds = (idx0, idx1)
        this_mask.view(B, -1)[fill_inds] = True
        rgb_features[fill_inds] = 0

        pos_inds = this_mask.view(B, -1).nonzero(as_tuple=True)[1]
        pos_indsn = this_mask.view(B, -1).count_nonzero(dim=-1)
        pos_indsn = torch.cat([torch.zeros(1, device=mask.device), torch.cumsum(pos_indsn, dim=0)])
        idx0 = torch.tensor([], device=mask.device, dtype=torch.int)
        idx1 = torch.tensor([], device=mask.device, dtype=torch.int)
        for i in range(B):
            offset = -diff[i].int().item()
            if offset > 0:
                pos_i = pos_indsn[i].int().item()
                idx0 = torch.cat((idx0, torch.full((offset,), i, device=mask.device)))
                idx1 = torch.cat((idx1, pos_inds[pos_i:pos_i + offset]))

        fill_inds = (idx0, idx1)
        this_mask.view(B, -1)[fill_inds] = False
        idx = this_mask.view(B, -1).nonzero(as_tuple=True)

        rgb_features = rgb_features[idx].reshape(B, n_sample, -1)

        pcd = pcd[idx].reshape(B, n_sample, -1)
        
        return rgb_features, pcd, idx

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def run_fps(self, context_features, context_pos):
        # context_features (B, Np, F)
        # context_pos (B, Np, F)
        # outputs of analogous shape, with smaller Np
        bs, npts, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
                context_features.to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            1,
            expanded_sampled_inds
        )

        # Sample positional embeddings
        _, _, ch = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        )
        sampled_context_pcd = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pcd


def test():
    from diffusion_policy.model.common.se3_util import random_se3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = DiffuserActorEncoder(
        backbone="clip", image_size=(128, 128), embedding_dim=60,
        num_sampling_level=3, nhist=3, num_attn_heads=3,
        fps_subsampling_factor=5
    ).to(device)



    rgb = torch.randn(64, 2, 3, 128, 128).to(device)
    pcd = torch.randn(64, 2, 3, 128, 128).to(device)
    rgb_feats_pyramid, pcd_pyramid = embedder.encode_images(rgb, pcd)
    print("RGB features pyramid shapes:")
    for i, rgb_feats in enumerate(rgb_feats_pyramid):
        print(f"Level {i}:", rgb_feats.shape)

    print("Point cloud pyramid shapes:")
    for i, pcd_i in enumerate(pcd_pyramid):
        print(f"Level {i}:", pcd_i.shape)

    curr_gripper = random_se3(64*3).reshape(64, 3, 4, 4).to(device)
    context_feats = torch.randn(64, 3, 60).to(device)
    context = torch.randn(64, 3, 3).to(device)
    
    curr_gripper_feats = embedder.encode_curr_gripper(
        curr_gripper, context_feats, context
    )

    pcd = pcd_pyramid[0]
    rgb_feats = einops.rearrange(
                rgb_feats_pyramid[0],
                "b ncam c h w -> b (ncam h w) c"
            )

    sampled_context_features, sampled_context_pcd = embedder.run_fps(
        rgb_feats, pcd
    )

    print("Sampled context features shape:", sampled_context_features.shape)
    print("Sampled context pcd shape:", sampled_context_pcd.shape)

if __name__ == "__main__":
    test()
    print("Test passed")