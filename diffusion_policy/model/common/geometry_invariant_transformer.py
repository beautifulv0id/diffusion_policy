import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math


from diffusion_policy.model.common.geometric_relative_attention import MultiHeadGeometricRelativeAttention


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


class GeometryInvariantTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 kv_dim=None,
                 return_last_attmap=False,
                 normal_attn_depth=0,
                 use_adaln=False,):

        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])
        self.normal_layers = nn.ModuleList([])

        linear_module_attn = lambda *args, **kwargs: JaxLinear(*args, **kwargs)
        linear_module_ff = lambda *args, **kwargs: ViTLinear(*args, **kwargs)

        prenorm_fn = lambda m: PreNorm(dim, m)
        for k in range(depth):
            attn = prenorm_fn(MultiHeadGeometricRelativeAttention(
                dim, heads=heads, dim_head=dim_head,
                dropout=dropout, kv_dim=kv_dim,
                linear_module=linear_module_attn, use_adaln=use_adaln))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.layers.append(nn.ModuleList([attn, ff]))

        for k in range(normal_attn_depth):
            attn = prenorm_fn(torch.nn.MultiheadAttention(
                embed_dim=dim, num_heads=heads, dropout=dropout, 
                kdim=kv_dim, vdim=kv_dim, batch_first=True))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.normal_layers.append(nn.ModuleList([attn, ff]))

        self.return_last_attmap = return_last_attmap

    def forward(self, x, z=None, extras=None, diff_ts=None):

        for l, (attn, ff) in enumerate(self.layers):
            if l == len(self.layers) - 1 and self.return_last_attmap and len(self.normal_layers) == 0:
                out, attmap = attn(x, z=z, return_attmap=True, diff_ts=diff_ts, extras=extras)
                x = out + x
            else:
                x = attn(x, z=z, extras=extras) + x
            x = ff(x) + x

        for l, (attn, ff) in enumerate(self.normal_layers):
            out, attmap = attn(x, key=z, value=z)
            x = out + x
            x = ff(x) + x

        if self.return_last_attmap:
            return x, attmap
        else:
            return x


############# MODEL TEST ##############
def geometric_relative_transformer_test():
    dim = 50
    context_dim = 64
    depth = 2
    heads = 3
    dim_head = 60
    mlp_dim = 256

    model = GeometryInvariantTransformer(dim, depth, heads, dim_head, mlp_dim, kv_dim=context_dim,
                        dropout=0.,
                        return_last_attmap=False)

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

    x = torch.randn(B, x_token, dim)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, x_token, 1, 1)

    z = torch.randn(B, z_token, context_dim)
    Xz = torch.randn(B, z_token, 3)

    extras = {'x_poses': Hx, 'z_poses': Xz,
              'x_types':'se3', 'z_types':'3D'}

    out = model(x, z=z, extras=extras)

    ############### TEST 2 #################
    rot = random_se3_batch(B)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Xz2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xz) + rot[:, None, :3, -1]

    extras = {'x_poses': Hx2, 'z_poses': Xz2,
              'x_types': 'se3', 'z_types': '3D'}
    out2 = model(x, z=z, extras=extras)

    print(torch.allclose(out, out2))
    print((out - out2).abs().max())


if __name__ == '__main__':

    geometric_relative_transformer_test()