import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention
from diffusion_policy.model.vision.resnet_wrapper import load_resnet50, load_resnet18
from diffusion_policy.model.vision.clip_wrapper import load_clip

from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformerEncoder, URSATransformer

class URSAFlowEncoder(ModuleAttrMixin):

    def get_res(self):
        return self.res

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=3,
                 num_attn_heads=8,
                 point_cloud_downsampling_factor=4,
                 fps_subsampling_factor=5,
                 quaternion_format='xyzw'):
        super().__init__()

        assert point_cloud_downsampling_factor in [2, 4, 8, 16]

        self.fps_subsampling_factor = fps_subsampling_factor
        self.point_cloud_downsampling_factor = point_cloud_downsampling_factor


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

        if point_cloud_downsampling_factor == 2:
            self.res = "res1"
            backbone_out_dim = 64
        elif point_cloud_downsampling_factor == 4:
            self.res = "res2"
            backbone_out_dim = 256
        elif point_cloud_downsampling_factor == 8:
            self.res = "res3"
            backbone_out_dim = 512
        elif point_cloud_downsampling_factor == 16:
            self.res = "res4"
            backbone_out_dim = 1024

        self.pcd_encoder = URSATransformerEncoder(
            d_model=embedding_dim, nhead=4, num_layers=2, args={'feature_type': 'fourier_and_distance'}
            )
        
        self.gripper_encoder = URSATransformer(
            d_model=embedding_dim, nhead=4, num_layers=2
        )

        self.to_out = nn.Conv2d(backbone_out_dim, embedding_dim, 1)


        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
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

        query = {'centers': gripper[:,:,:3,3], 'vectors': gripper[:,:,:3,:3]}
        key = {'centers': context, 'vectors': torch.zeros(context.shape + (3,)).to(context.device)}

        # Pass gripper through transformer
        gripper_feats = self.gripper_encoder(
            tgt=gripper_feats, memory=context_feats,
            geometric_args={'query': query, 'key': key}
        )

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
        rgb_features = rgb_features[self.res]
        rgb_features = self.to_out(rgb_features)

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
            "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
        )

        return rgb_features, pcd
    
    def encode_pcd(self, pcd, feats):
        """
        Encode point cloud data

        Args:
            - rgb_feats: (B, npt, C)
            - pcd: (B, npt, 3)

        Returns:
            - pcd_feats: (B, npt, F)
        """
        batch = pcd.shape[0]
        npt = pcd.shape[1]
        
        points_x = {'centers': pcd, 'vectors': torch.zeros(batch, npt, 3, 3).to(pcd.device)}
        pcd_feats = self.pcd_encoder(tgt=feats, geometric_args={'query': points_x})

        return pcd_feats



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
    from diffusion_policy.common.se3_util import random_se3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = URSAFlowEncoder(
        backbone="clip", embedding_dim=60,
        nhist=3, num_attn_heads=3,
        fps_subsampling_factor=5,
        point_cloud_downsampling_factor=2
    ).to(device)


    batch = 8
    rgb = torch.randn(batch, 4, 3, 128, 128).to(device)
    pcd = torch.randn(batch, 4, 3, 128, 128).to(device)
    curr_gripper = random_se3(batch*3).reshape(batch, 3, 4, 4).to(device)

    rgb_feats, pcd = embedder.encode_images(rgb, pcd)
    rgb_feats = einops.rearrange(
                rgb_feats,
                "b ncam c h w -> b (ncam h w) c"
            )

    # pcd_feats = embedder.encode_pcd(pcd, rgb_feats)
    
    curr_gripper_feats = embedder.encode_curr_gripper(
        curr_gripper, rgb_feats, pcd
    )

    sampled_context_features, sampled_context_pcd = embedder.run_fps(
        rgb_feats, pcd
    )

    print("Sampled context features shape:", sampled_context_features.shape)
    print("Sampled context pcd shape:", sampled_context_pcd.shape)

if __name__ == "__main__":
    test()
    print("Test passed")