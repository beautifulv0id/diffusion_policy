import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, steps=100):
        super().__init__()
        self.dim = dim
        self.steps = steps

    def forward(self, x):
        x = self.steps * x
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FeaturePCloudPolicy(nn.Module):
    def __init__(self, dim=100, depth=1, heads=3, dim_head=64, mlp_dim=256,
                       context_dim=100, n_context=10):

        super().__init__()

        ## Time-Context Embedding ##
        self.time_encoder = SinusoidalPosEmb(dim)

        ## Input Embeddings ##
        self.x_embeddings = nn.Parameter(torch.randn(1, dim))
        self.context_embeddings = nn.Parameter(torch.randn(n_context, dim))

        ## Geometry Invariant Transformer ##
        self.git = GeometryInvariantTransformer(dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 kv_dim=context_dim,
                 return_last_attmap=False)

        ## To-Output ##
        self.to_out = nn.Linear(dim, 6)

    def set_context(self, context_pose, context_features=None):

        self.context_x = context_pose
        if context_features is None:
            self.context_features = self.context_embeddings[None,...].repeat(context_pose.shape[0],1,1)
        else:
            self.context_features = context_features

    def forward(self, x, t):
        time_embeddings = self.time_encoder(t)
        x_features = self.x_embeddings[None,...].repeat(x.shape[0],1,1)
        x_features = x_features + time_embeddings[:,None,:]
        _context_features = self.context_features + time_embeddings[:,None,:]

        extras = {'x_poses': x[:,None,...], 'z_poses': self.context_x,
                  'x_types': 'se3', 'z_types': '3D'}

        out = self.git(x_features, z=_context_features, extras=extras)

        return self.to_out(out).squeeze(1)


def feature_pcloud_git_test():
    n_context_tokens = 9
    model = FeaturePCloudPolicy(n_context=n_context_tokens)

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

    Hx = random_se3_batch(B)
    t = torch.rand(B)
    Xz = torch.randn(B, n_context_tokens, 3)

    model.set_context(context_pose=Xz)
    out = model(Hx, t)


    ############### TEST 2 #################
    rot = random_se3_batch(B)
    Hx2 = torch.einsum('bmn,bnk->bmk', rot, Hx)
    Xz2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xz) + rot[:, None, :3, -1]

    extras = {'x_poses': Hx2, 'z_poses': Xz2,
              'x_types': 'se3', 'z_types': '3D'}
    model.set_context(context_pose=Xz2)
    out2 = model(Hx2, t)

    print(torch.allclose(out, out2, rtol=1e-05, atol=1e-05))
    print((out - out2).abs().max())


if __name__ == '__main__':

    feature_pcloud_git_test()

