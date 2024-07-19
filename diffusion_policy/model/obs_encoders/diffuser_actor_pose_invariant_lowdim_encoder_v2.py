import dgl.geometry as dgl_geo
import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding9D
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule
from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention
from diffusion_policy.common.rotation_utils import normalise_quat
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

class DiffuserActorEncoder(ModuleAttrMixin):

    def __init__(self,
                 embedding_dim=60,
                 nhist=3,
                 nkeypoints=10,
                 num_attn_heads=8,
                 fps_subsampling_factor=5,
                 quaternion_format='xyzw'):
        super().__init__()

        self.fps_subsampling_factor = fps_subsampling_factor

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding9D(embedding_dim)

        head_dim = embedding_dim // num_attn_heads
        assert head_dim * num_attn_heads == embedding_dim, "embed_dim must be divisible by num_heads"

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
 
        self.gripper_context_head = InvariantPointTransformer(
            embedding_dim, 3, num_attn_heads, head_dim, kv_dim=embedding_dim, use_adaln=False,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3)

        # Goal gripper learnable features
        self.goal_gripper_embed = nn.Embedding(1, embedding_dim)

        # Keypoint embeddings
        self.keypoint_embeddings = nn.Embedding(nkeypoints, embedding_dim)

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


    def get_keypoint_embeddings(self):
        return self.keypoint_embeddings.weight    

def test():
    from diffusion_policy.model.common.so3_util import random_so3
    from diffusion_policy.model.common.se3_util import random_se3

    embedding_dim = 192
    num_attn_heads = 8
    encoder = DiffuserActorEncoder(
        embedding_dim=embedding_dim,
        num_attn_heads=num_attn_heads
    )
    pcd = torch.randn(10, 10, 3)
    context_feats = torch.randn(10, 10, embedding_dim)
    gripper = random_se3(10*3).reshape(10, 3, 4, 4)

    gripper_feats = encoder.encode_curr_gripper(gripper, context_feats, pcd)

    # translate frame
    trans = torch.randn(10, 3)
    pcd2 = trans.unsqueeze(1) + pcd
    context_feats2 = context_feats
    gripper2 = gripper.clone()
    gripper2[..., :3, 3] = gripper2[...,:3, 3] + trans.unsqueeze(1)
    gripper_feats2 = encoder.encode_curr_gripper(gripper2, context_feats2, pcd2)
    assert torch.allclose(gripper_feats, gripper_feats2, rtol=1e-5, atol=1e-5), "Frame translation failed"

    # rotate frame
    rot = quaternion_to_matrix(torch.randn(10, 4))
    pcd3 = torch.einsum('bmn,btn->btm', rot, pcd)
    context_feats2 = context_feats
    gripper3 = gripper.clone()
    gripper3[...,:3,:4] = torch.einsum('bmn,btnk->btmk', rot, gripper[...,:3,:4])
    gripper_feats3 = encoder.encode_curr_gripper(gripper3, context_feats2, pcd3)
    assert torch.allclose(gripper_feats, gripper_feats3, rtol=1e-5, atol=1e-5), "Frame rotation failed"
    print("Success")


if __name__ == '__main__':
    test()