import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange


__USE_DEFAULT_INIT__ = False

class AdaLN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (B, N, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x  # (B, N, C)

def invert_se3(matrix):
    """
    Invert a homogeneous transformation matrix.

    :param matrix: A 4x4 numpy array representing a homogeneous transformation matrix.
    :return: The inverted transformation matrix.
    """


    # Extract rotation (R) and translation (t) from the matrix
    R = matrix[...,:3, :3]
    t = matrix[...,:3, 3]

    # Invert the rotation (R^T) and translation (-R^T * t)
    R_inv = torch.transpose(R, -1, -2)
    t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)

    # Construct the inverted matrix
    inverted_matrix = torch.clone(matrix)
    inverted_matrix[...,:3, :3] = R_inv
    inverted_matrix[...,:3, 3] = t_inv

    return inverted_matrix


class AttnFn(torch.nn.Module):
    def __init__(self, scale):
        self.scale = scale
        super().__init__()

    def forward(self, q, k, v):
        sim = q @ k.transpose(-1, -2)  # [B, H, Nq*Tq, Nk*Tk]
        attn = nn.Softmax(-1)(sim * self.scale)
        out = (attn @ v)
        return out, attn

def _add_se3(v, x, query=True):
    B, H, N, D = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    _v = v.reshape(B, H, N, -1, x.shape[-1])
    if query:
        return torch.einsum('bnij,bhncj->bhnci', x.transpose(-1,-2), _v).reshape(B,H,N,D)
    else:
        return torch.einsum('bnij,bhncj->bhnci', x, _v).reshape(B,H,N,D)

def _add_3d(v, x):
    B, H, N, D = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    _v = v.reshape(B, H, N, -1, x.shape[-1])
    return torch.einsum('bnj,bhncj->bhncj', x, _v).reshape(B, H, N, D)

def q_rel_map(v, x):
    return _add_se3(v, x, query=True)

def kv_rel_map(v, x):
    return _add_se3(v, x, query=False)

class GeomRelAttn(torch.nn.Module):
    def __init__(self, attn_fn):
        super().__init__()
        self.attn_fn = attn_fn

    def set_relative_poses(self, q_poses, k_poses, q_types='se3', k_types='3D'):
        """
        Args:
            q_poses: Tensor of shape [B, Nq, D] where Nq is the num of query tokens and D the dimension
            k_pose: Tensor of shape [B, Nk, D] where Nk is the num of key-value tokens and D the dimension
            D = (2D:2, SO(2):2x2, SE(2):3x3, 3D:3, SO(3):3x3, SE(3): 4x4)
            q_type: Tensor of shape [B, Nq] with the type of object
            k_type: Tensor of shape [B, Nq] with the type of object
        """
        
        if q_types is not dict:
            if q_types == 'se3':
                self.q_inv_poses = invert_se3(q_poses)
                self.q_rel_map = q_rel_map
                self.out_rel_map = kv_rel_map

        if k_types is not dict:
            if k_types == 'se3':
                self.kv_poses = k_poses
                self.kv_rel_map = kv_rel_map

            if k_types == '3D':
                _k_poses = torch.cat((k_poses, torch.ones_like(k_poses[..., :1])), dim=-1)
                self.kv_poses = _k_poses[...,None].repeat(1,1,1,4)

                # inverted_poses = torch.einsum('bmn,btnl->btml',self.q_inv_poses[:,0,...],self.kv_poses)
                # self.kv_poses = inverted_poses
                # self.q_inv_poses = torch.eye(4)[None,None, ...].repeat(self.q_inv_poses.shape[0], self.q_inv_poses.shape[1],1,1)

                self.kv_rel_map = kv_rel_map
                #self.kv_rel_map = _add_3d

        # previous_x = torch.einsum('btmn,bln->btlm',self.q_inv_poses, self.kv_poses)
        # self.previous_x = previous_x

    def _add_position_to_qkv(self, q, k ,v, v_transform=True):
        qx = self.q_rel_map(q, self.q_inv_poses)
        kx = self.kv_rel_map(k, self.kv_poses)
        if v_transform:
            vx = self.kv_rel_map(v, self.kv_poses)
        else:
            vx = v
        return qx, kx, vx

    def _set_relative_output(self, out, v_transform=True):
        if v_transform:
            return self.out_rel_map(out, self.q_inv_poses)
        else:
            return out

    def forward(self, q, k, v,
                v_transform=False):
        """
        Args:
            q: Tensor of shape [B, H, Nq, C] where Nq is the num of query tokens
            k: Tensor of shape [B, H, Nk, C] where Nk is the num of key-value tokens
            v: Tensor of shape [B, H, Nk, Cv]
            attn_fn: Attnetion function that outpus f(QK) given QK. f is for example Softmax in vanilla transformer.
            reps: Dict contains pre-computed representation matrices ρ
            trans_coeff: Scaler that adjusts scale coeffs

        Return:
            Tensor of shape [B, H, Nk, Cv]
        """

        B, H, Nq, C = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        Nk = k.shape[2]

        ## Add Global Pose to Q,K,V
        q,k,v = self._add_position_to_qkv(q, k, v, v_transform)

        out, attn = self.attn_fn(q, k, v)

        out = self._set_relative_output(out, v_transform)
        return out, attn


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


class MultiHeadGeometricRelativeAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None,
                 linear_module=JaxLinear, use_bias = True, v_transform=True, 
                 use_adaln=False,
                 **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # kv_dim being None indicates this attention is used as self-attention
        self.selfatt = (kv_dim is None)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_bias = use_bias
        self.v_transform = v_transform

        ## Define Attention Network ##
        attn_fn = AttnFn(scale=self.scale)
        self.geometric_attention = GeomRelAttn(attn_fn=attn_fn)

        # parse
        q_inner_dim = inner_dim
        if kv_dim is not None:
            self.to_q = linear_module(dim, q_inner_dim, bias=self.use_bias)
            self.to_kv = linear_module(
                kv_dim, 2 * inner_dim, bias=self.use_bias)
        else:
            self.to_qkv = linear_module(
                dim, inner_dim * 2 + q_inner_dim, bias=self.use_bias)

        self.to_out = nn.Sequential(
            linear_module(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if use_adaln:
            self.adaln = AdaLN(dim)

    def forward(self, x, z=None, return_attmap=False, diff_ts=None, extras={}):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        else:
            x = x

        ## Set Positional Embeddings ##
        if self.selfatt:
            q_poses, k_poses, q_types, k_types = extras['x_poses'], extras['x_poses'], extras['x_types'], extras['x_types']
        else:
            q_poses, k_poses, q_types, k_types = extras['x_poses'], extras['z_poses'], extras['x_types'], extras['z_types']
        self.geometric_attention.set_relative_poses(q_poses=q_poses, k_poses=k_poses, q_types=q_types, k_types=k_types)

        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        out, attn = self.geometric_attention(q, k, v, v_transform=self.v_transform)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attmap:
            return out, attn
        else:
            return out



######################## TEST OF THE MODELS ########################
def geometric_relative_attention_test():
    v_transform = True
    B = 150

    H = 10
    q_dim = 120
    k_dim = 120
    v_dim = 120

    q_token = 3
    k_token = 30

    ## Attention Function ##
    attn_fn = AttnFn(scale=1.)
    multi_gra = GeomRelAttn(attn_fn=attn_fn)

    q = torch.randn(B, H, q_token, q_dim)

    Hq = random_se3_batch(B)[:, None, ...].repeat(1, q_token, 1, 1)

    k = torch.randn(B, H, k_token, k_dim)
    v = torch.randn(B, H, k_token, v_dim)
    Xk = torch.randn(B, k_token, 3)

    multi_gra.set_relative_poses(q_poses=Hq, k_poses=Xk)
    out = multi_gra(q, k, v, v_transform=v_transform)

    rot = random_se3_batch(B)
    Hq2 = torch.einsum('bmn,btnk->btmk', rot, Hq)
    Xk2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xk) + rot[:, None, :3, -1]

    multi_gra.set_relative_poses(q_poses=Hq2, k_poses=Xk2)
    out2 = multi_gra(q, k, v, v_transform=v_transform)

    print(torch.allclose(out[0], out2[0]))
    print((out[0] - out2[0]).abs().max())

    rot = random_se3_batch(B)
    ones = torch.eye(3)[None, ...].repeat(B, 1, 1)
    rot[:, :3, :3] = ones
    Hq2 = torch.einsum('bmn,btnk->btmk', rot, Hq)
    Xk2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xk) + rot[:, None, :3, -1]

    multi_gra.set_relative_poses(q_poses=Hq2, k_poses=Xk2)
    out2 = multi_gra(q, k, v, v_transform=v_transform)

    print(torch.allclose(out[0], out2[0]))
    print((out[0] - out2[0]).abs().max())


def multihead_geometric_attention_test():

    B = 150

    H = 10
    q_dim = 56
    k_dim = 12

    q_token = 3
    k_token = 30

    ## Attention Function ##
    multihead_attention = MultiHeadGeometricRelativeAttention(dim=q_dim, heads=H, dim_head=64, kv_dim=k_dim)

    ## Run test ##
    x = torch.randn(B, q_token, q_dim)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, q_token, 1, 1)

    z = torch.randn(B, k_token, k_dim)
    Xz = torch.randn(B, k_token, 3)
    extras = {'x_poses': Hx, 'z_poses': Xz,
              'x_types':'se3', 'z_types':'3D'}

    out = multihead_attention(x, z=z, extras=extras)

    ############### TEST 2 #################
    rot = random_se3_batch(B)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Xz2 = torch.einsum('bmn,btn->btm', rot[..., :3, :3], Xz) + rot[:, None, :3, -1]

    extras = {'x_poses': Hx2, 'z_poses': Xz2,
              'x_types': 'se3', 'z_types': '3D'}
    out2 = multihead_attention(x, z=z, extras=extras)

    print(torch.allclose(out, out2))
    print((out - out2).abs().max())


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
    H = torch.eye(4).unsqueeze(0).repeat(batch_size,1,1)
    H[:, :3, :3] = random_so3_batch(batch_size)
    H[:, :3, 3] = torch.randn(batch_size, 3)
    return H


if __name__ == "__main__":
    geometric_relative_attention_test()

    multihead_geometric_attention_test()



