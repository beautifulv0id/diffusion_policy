import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math


from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPoseAttention, InvariantPointAttention


__USE_DEFAULT_INIT__ = False


class JaxLinear(nn.Linear):
    """ Linear layers with initialization matching the Jax defaults """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            input_size = self.weight.shape[-1]
            std = math.sqrt(1 / input_size)
            init.trunc_normal_(self.weight, std=std, a=-2. * std, b=2. * std)
            if self.bias is not None:
                init.zeros_(self.bias)


class ViTLinear(nn.Linear):
    """ Initialization for linear layers used by ViT """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std=1e-6)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(
            dim) if dim is not None else lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., linear_module=ViTLinear):
        super().__init__()
        self.net = nn.Sequential(
            linear_module(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            linear_module(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class InvariantPoseTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=256,
                 dropout=0.,
                 kv_dim=None,
                 attention_module=InvariantPoseAttention):

        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])

        linear_module_ff = lambda *args, **kwargs: ViTLinear(*args, **kwargs)

        prenorm_fn = lambda m: PreNorm(dim, m)
        for k in range(depth):
            attn = prenorm_fn(attention_module(
                dim, heads=heads, dim_head=dim_head,
                dropout=dropout, kv_dim=kv_dim))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.layers.append(nn.ModuleList([attn, ff]))

    def forward(self, x, poses_x, z=None, poses_z=None, mask=None):

        for l, (attn, ff) in enumerate(self.layers):
            x = attn(x, z=z, poses_x=poses_x, poses_z=poses_z, mask=mask) + x
            x = ff(x) + x

        return x


############# MODEL TEST ##############
def geometric_relative_transformer_test():

    import time

    dim = 50
    context_dim = 64
    depth = 5
    heads = 3
    dim_head = 60
    mlp_dim = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InvariantPoseTransformer(dim, depth, heads, dim_head, mlp_dim, kv_dim=context_dim,
                                    dropout=0., attention_module=InvariantPoseAttention).to(device)

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