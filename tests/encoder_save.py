import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
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
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5):
        super().__init__()
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [(128, 128), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor

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

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Instruction encoder
        self.instruction_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

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

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.goal_gripper_embed,
            context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
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

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

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
        idx = this_mask.view(B, -1).nonzero(as_tuple=True)

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

        rgb_features = einops.rearrange(rgb_features, 'b v c h w -> b (v h w) c')
        rgb_features = rgb_features[idx].reshape(B, n_sample, -1)

        pcd = pcd[idx].reshape(B, n_sample, -1)

        return rgb_features, pcd



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

    def encode_instruction(self, instruction):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instruction: (B, max_instruction_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instruction_encoder(instruction)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instruction), instr_feats.shape[1], 3,
            device=instruction.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    


def test():
    embedder = DiffuserActorEncoder(
        backbone="clip", image_size=(256, 256), embedding_dim=60,
        num_sampling_level=3, nhist=3, num_attn_heads=3,
        num_vis_ins_attn_layers=2, fps_subsampling_factor=5
    )

    rgb = torch.randn(64, 2, 3, 256, 256)
    pcd = torch.randn(64, 2, 3, 256, 256)
    rgb_feats_pyramid, pcd_pyramid = embedder.encode_images(rgb, pcd)
    print("RGB features pyramid shapes:")
    for i, rgb_feats in enumerate(rgb_feats_pyramid):
        print(f"Level {i}:", rgb_feats.shape)

    print("Point cloud pyramid shapes:")
    for i, pcd_i in enumerate(pcd_pyramid):
        print(f"Level {i}:", pcd_i.shape)

    curr_gripper = torch.randn(64, 3, 3)
    context_feats = torch.randn(64, 3, 60)
    context = torch.randn(64, 3, 3)
    
    curr_gripper_feats, curr_gripper_pos = embedder.encode_curr_gripper(
        curr_gripper, context_feats, context
    )

if __name__ == "__main__":
    test()
    print("Test passed")