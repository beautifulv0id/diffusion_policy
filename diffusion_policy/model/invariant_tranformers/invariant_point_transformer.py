import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPoseAttention, InvariantPointAttention

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)


class IPABlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        dropout=0.,
        kv_dim=None,
        ff_mult = 1,
        ff_num_layers = 3,          # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,           # in the paper, they used post-layernorm - offering pre-norm as well
        post_attn_dropout = 0.,
        post_ff_dropout = 0.,
        attention_module = InvariantPointAttention,
        point_dim = 4,
        use_adaln = False,
        **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = attention_module(dim, heads=heads, dim_head=dim_head,
                dropout=dropout, kv_dim=kv_dim, point_dim=point_dim, use_adaln=use_adaln)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, x, z, poses_x, poses_z, mask = None, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, poses_x=poses_x, z=z, poses_z=poses_z, mask=mask, **kwargs) + x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        x = self.ff_norm(x) if post_norm else x
        return x

# add an IPA Transformer - iteratively updating rotations and translations

# this portion is not accurate to AF2, as AF2 applies a FAPE auxiliary loss on each layer, as well as a stop gradient on the rotations
# just an attempt to see if this could evolve to something more generally usable

class InvariantPointTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads, dim_head,
        kv_dim=None,
        dropout=0.,
        attention_module=InvariantPointAttention,
        point_dim=4,
        use_adaln=False,
        **kwargs
    ):
        super().__init__()


        # layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                IPABlock(dim=dim, heads = heads,dim_head = dim_head,
                         dropout = dropout, kv_dim=kv_dim,
                         attention_module=attention_module,
                         point_dim=point_dim, use_adaln=use_adaln))

        # whether to detach rotations or not, for stability during training
        self.to_points = nn.Linear(dim, dim)

    def forward(self, x, poses_x, z=None, poses_z=None, mask=None, point_mask_x=None, point_mask_z=None, diff_ts=None):

        # go through the layers and apply invariant point attention and feedforward
        for block in self.layers:
            x = block(x, z=z, poses_x=poses_x, poses_z=poses_z, mask=mask, point_mask_x=point_mask_x, point_mask_z=point_mask_z, diff_ts=diff_ts)

        return self.to_points(x)


############# MODEL TEST ##############
def geometric_relative_transformer_test():

    import time

    dim = 50
    context_dim = 64
    depth = 5
    heads = 3
    dim_head = 60

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InvariantPointTransformer(dim, depth, heads, dim_head, kv_dim=context_dim,
                        dropout=0., attention_module=InvariantPointAttention).to(device)

    ############### SAMPLE ################

    def random_so3_batch(batch_size):
        """
        Generate a batch of random SO(3) rotation matrices using quaternions.

        :param batch_size: Number of rotation matrices to generate.
        :return: A tensor of shape (batch_size, 3, 3) containing SO(3) rotation matrices.
        """
        # Generate random quaternions
        q = torch.randn(batch_size, 4)
        q = torch.nn.functional.normalize(q, p=2, dim=1)

        # Convert quaternions to rotation matrices
        rotation_matrices = torch.zeros(batch_size, 3, 3)

        # Quaternion components
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # First row of the rotation matrix
        rotation_matrices[:, 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
        rotation_matrices[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
        rotation_matrices[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        rotation_matrices[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
        rotation_matrices[:, 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
        rotation_matrices[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        rotation_matrices[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
        rotation_matrices[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
        rotation_matrices[:, 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)

        return rotation_matrices

    def random_se3_batch(batch_size):
        '''
        This function creates SE(3) homogeneous matrices
        :param batch_size: N
        :return: Nx4x4 tensor of shape (batch_size, 4, 4)
        '''
        H = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        H[:, :3, :3] = random_so3_batch(batch_size)
        H[:, :3, 3] = torch.randn(batch_size, 3)
        return H

    B = 120
    x_token = 2
    z_token = 12

    x = torch.randn(B, x_token, dim).to(device)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, x_token, 1, 1).to(device)

    z = torch.randn(B, z_token, context_dim).to(device)
    Hz = random_se3_batch(B)[:, None, ...].repeat(1, z_token, 1, 1).to(device)

    poses_x = {'rotations': Hx[..., :3, :3], 'translations': Hx[..., :3, -1], 'types': 'se3'}
    poses_z = {'rotations': Hz[..., :3, :3], 'translations': Hz[..., :3, -1], 'types': 'se3'}

    out = model(x, z=z, poses_x=poses_x, poses_z=poses_z)

    ############### TEST 2 #################
    rot = random_se3_batch(B).to(device)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Hz2 = torch.einsum('bmn,btnk->btmk', rot, Hz)

    poses_x = {'rotations': Hx2[..., :3, :3], 'translations': Hx2[..., :3, -1], 'types': 'se3'}
    poses_z = {'rotations': Hz2[..., :3, :3], 'translations': Hz2[..., :3, -1], 'types': 'se3'}

    time0 = time.time()
    out2 = model(x, z=z, poses_x=poses_x, poses_z=poses_z)
    print('Time: ', time.time() - time0)

    print('Is it invariant? ', torch.allclose(out, out2, atol=1e-5))
    print((out - out2).abs().max())


if __name__ == '__main__':

    geometric_relative_transformer_test()