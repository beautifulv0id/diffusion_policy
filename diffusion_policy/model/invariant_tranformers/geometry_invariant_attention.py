import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from contextlib import contextmanager

from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb


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


def invert_se3(rotations, translations):
    """
    Invert a homogeneous transformation matrix.

    :param matrix: A 4x4 numpy array representing a homogeneous transformation matrix.
    :return: The inverted transformation matrix.
    """


    # Extract rotation (R) and translation (t) from the matrix
    R = rotations
    t = translations

    # Invert the rotation (R^T) and translation (-R^T * t)
    R_inv = torch.transpose(R, -1, -2)
    t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)

    # Construct the inverted matrix
    inverted_matrix = torch.eye(4, device=rotations.device)[None,None,...].repeat(rotations.shape[0], rotations.shape[1], 1, 1)
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


class InvariantPoseAttentionFn(torch.nn.Module):
    def __init__(self, attn_fn):
        super().__init__()
        self.attn_fn = attn_fn

    def set_relative_poses(self, x_rotations, x_translations, z_rotations, z_translations):
        """
        Args:
            q_poses: Tensor of shape [B, Nq, D] where Nq is the num of query tokens and D the dimension
            k_pose: Tensor of shape [B, Nk, D] where Nk is the num of key-value tokens and D the dimension
            D = (2D:2, SO(2):2x2, SE(2):3x3, 3D:3, SO(3):3x3, SE(3): 4x4)
            q_type: Tensor of shape [B, Nq] with the type of object
            k_type: Tensor of shape [B, Nq] with the type of object
        """
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

        ## Set Query Poses ##
        if x_rotations is not None:
            self.q_inv_poses = invert_se3(x_rotations, x_translations)
            self.q_rel_map = lambda v,x: _add_se3(v, x, query=True)
            self.out_rel_map = lambda v,x: _add_se3(v, x, query=False)

        ## Set Key-Value Poses ##
        if z_rotations is not None:
            self.kv_poses = torch.eye(4, device=z_rotations.device)[None,None,...].repeat(z_rotations.shape[0], z_rotations.shape[1], 1, 1)
            self.kv_poses[..., :3, :3] = z_rotations
            self.kv_poses[..., :3, -1] = z_translations
            self.kv_rel_map = lambda v,x: _add_se3(v, x, query=False)

        else:
            _k_poses = torch.cat((z_translations, torch.ones_like(z_translations[..., :1])), dim=-1)
            self.kv_poses = _k_poses[...,None].repeat(1,1,1,4)
            self.kv_rel_map = lambda v, x: _add_se3(v, x, query=False)

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


class InvariantPoseAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None,
                 linear_module=JaxLinear, use_bias = True, v_transform=True,
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
        self.geometric_attention = InvariantPoseAttentionFn(attn_fn=attn_fn)

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

    def forward(self, x, poses_x, z=None, poses_z=None, mask=None):

        ## Set Positional Embeddings ##
        if z is None:
            z_rotations, z_translations = poses_x['rotations'], poses_x['translations']
        else:
            z_rotations, z_translations = poses_z['rotations'], poses_z['translations']

        self.geometric_attention.set_relative_poses(x_rotations=poses_x['rotations'], x_translations=poses_x['translations'],
                                                    z_rotations=z_rotations, z_translations=z_translations)

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

        return out



###################### POINT INVARIANT ATTENTION ######################
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value


class InvariantPointAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None,
                 point_dim=4, eps=1e-8, use_adaln=False, **kwargs):

        super().__init__()
        if kv_dim is None:
            kv_dim = dim

        self.eps = eps
        self.heads = heads

        # num attention contributions
        num_attn_logits = 2

        # qkv projection for scalar attention (normal)
        self.scalar_attn_logits_scale = (num_attn_logits * dim_head) ** -0.5

        self.to_scalar_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_scalar_k = nn.Linear(kv_dim, dim_head * heads, bias = False)
        self.to_scalar_v = nn.Linear(kv_dim, dim_head * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(kv_dim, point_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(kv_dim, point_dim * heads * 3, bias = False)


        # ## Point distance to features ##
        # self.point_distance_encoder = nn.Sequential(
        #     nn.Linear(point_dim, point_dim*4),
        #     nn.Mish(),
        #     nn.Linear(point_dim*4, 1)
        # )

        # pairwise representation projection to attention bias
        pairwise_repr_dim = 0


        # combine out - scalar dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        self.to_out = nn.Linear(heads * (dim_head + pairwise_repr_dim + point_dim * (3 + 1)), dim)

        if use_adaln:
            self.adaln = AdaLN(dim)

    def forward(self, x, poses_x, z=None, poses_z=None, point_mask_x=None, point_mask_z=None,  mask=None, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        else:
            x = x

        x, b, h, eps = x, x.shape[0], self.heads, self.eps
        if z is None:
            z=x
        if poses_z is None:
            poses_z = poses_x

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(z), self.to_scalar_v(z)

        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(z), self.to_point_v(z)

        if point_mask_x is not None:
            q_point[point_mask_x] = 0
        if point_mask_z is not None:
            k_point[point_mask_z] = 0
            v_point[point_mask_z] = 0

        # split out heads
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))
        
        ## Extract Poses ##
        rotations_q = repeat(poses_x['rotations'], 'b n r1 r2 -> (b h) n r1 r2', h = h)
        translations_q = repeat(poses_x['translations'], 'b n c -> (b h) n () c', h = h)
        rotations_kv = repeat(poses_z['rotations'], 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations_kv = repeat(poses_z['translations'], 'b n c -> (b h) n () c', h=h)

        # rotate qkv points into global frame
        q_point = einsum('b n c r, b n d r -> b n d c', rotations_q, q_point) + translations_q
        k_point = einsum('b n c r, b n d r -> b n d c', rotations_kv, k_point) + translations_kv
        v_point = einsum('b n c r, b n d r -> b n d c', rotations_kv, v_point) + translations_kv

        # derive attn logits for scalar and pairwise
        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale

        # derive attn logits for point attention
        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = (-1, -2))
        #point_dist = self.point_distance_encoder(point_dist).squeeze(-1)
        #point_dist = point_dist.mean(dim=-1)

        # self.point_qk_diff = point_qk_diff

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_points


        # mask

        if exists(mask):
            #mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask[None,...], 'b i j -> (b h) i j', h = attn_logits.shape[0])
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim = - 1)

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision

            # aggregate values

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            # if require_pairwise_repr:
            #     results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum('b n c r, b n d r -> b n d c', rotations_q.transpose(-1, -2), results_points - translations_q[:,:,0,None,:])
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h = h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h = h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h = h)

        results = (results_scalar, results_points, results_points_norm)

        # if require_pairwise_repr:
        #     results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h = h)
        #     results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim = -1)
        return self.to_out(results)


######################## TEST OF THE MODELS ########################

def pose_invariant_attention_test():

    B = 150

    H = 10
    q_dim = 56
    k_dim = 12

    q_token = 3
    k_token = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Attention Function ##
    multihead_attention = InvariantPoseAttention(dim=q_dim, heads=H, dim_head=64, kv_dim=k_dim).to(device)

    ## Run test ##
    x = torch.randn(B, q_token, q_dim).to(device)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, q_token, 1, 1).to(device)

    z = torch.randn(B, k_token, k_dim).to(device)
    Hz = random_se3_batch(B)[:, None, ...].repeat(1, k_token, 1, 1).to(device)

    poses_x = {'rotations': Hx[..., :3, :3], 'translations': Hx[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz[..., :3, :3], 'translations': Hz[..., :3, -1], 'types':'se3'}

    out = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)
    ############### TEST 2 #################
    rot = random_se3_batch(B).to(device)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Hz2 = torch.einsum('bmn,btnk->btmk', rot, Hz)

    poses_x = {'rotations': Hx2[..., :3, :3], 'translations': Hx2[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz2[..., :3, :3], 'translations': Hz2[..., :3, -1], 'types':'se3'}

    time0 = time.time()
    out2 = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)
    print('Time: ', time.time() - time0)

    print('Is it invariant? ',torch.allclose(out, out2, atol=1e-5))
    print((out - out2).abs().max())


def point_invariant_attention_test():

    B = 150

    H = 10
    q_dim = 56
    k_dim = 12

    q_token = 11
    k_token = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Attention Function ##
    multihead_attention = InvariantPointAttention(dim=q_dim, heads=H, dim_head=64, kv_dim=k_dim).to(device)

    ## Run test ##
    x = torch.randn(B, q_token, q_dim).to(device)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, q_token, 1, 1).to(device)

    z = torch.randn(B, k_token, k_dim).to(device)
    Hz = random_se3_batch(B)[:, None, ...].repeat(1, k_token, 1, 1).to(device)

    poses_x = {'rotations': Hx[..., :3, :3], 'translations': Hx[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz[..., :3, :3], 'translations': Hz[..., :3, -1], 'types':'se3'}

    time0 = time.time()
    out = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)
    print('Time: ', time.time() - time0)

    ############### TEST 2 #################
    rot = random_se3_batch(B).to(device)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Hz2 = torch.einsum('bmn,btnk->btmk', rot, Hz)

    poses_x = {'rotations': Hx2[..., :3, :3], 'translations': Hx2[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz2[..., :3, :3], 'translations': Hz2[..., :3, -1], 'types':'se3'}

    out2 = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)

    print('Is it invariant? ',torch.allclose(out, out2, atol=1e-5))
    print((out - out2).abs().max())

def point_invariant_attention_v_2_test():

    B = 150

    H = 10
    q_dim = 56
    k_dim = 12

    q_token = 11
    k_token = 32


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q_mask = torch.randint(0, 2, (B, q_token)).bool().to(device)
    k_mask = torch.randint(0, 2, (B, k_token)).bool().to(device)
    ## Attention Function ##
    multihead_attention = InvariantPointAttention(dim=q_dim, heads=H, dim_head=64, kv_dim=k_dim).to(device)

    ## Run test ##
    x = torch.randn(B, q_token, q_dim).to(device)
    Hx = random_se3_batch(B)[:, None, ...].repeat(1, q_token, 1, 1).to(device)

    z = torch.randn(B, k_token, k_dim).to(device)
    Hz = random_se3_batch(B)[:, None, ...].repeat(1, k_token, 1, 1).to(device)

    poses_x = {'rotations': Hx[..., :3, :3], 'translations': Hx[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz[..., :3, :3], 'translations': Hz[..., :3, -1], 'types':'se3'}

    poses_x['rotations'][q_mask] = torch.eye(3).to(device)
    poses_z['rotations'][k_mask] = torch.eye(3).to(device)

    time0 = time.time()
    out = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)
    print('Time: ', time.time() - time0)

    ############### TEST 2 #################
    rot = random_se3_batch(B).to(device)
    Hx2 = torch.einsum('bmn,btnk->btmk', rot, Hx)
    Hz2 = torch.einsum('bmn,btnk->btmk', rot, Hz)

    poses_x = {'rotations': Hx2[..., :3, :3], 'translations': Hx2[..., :3, -1], 'types':'se3'}
    poses_z = {'rotations': Hz2[..., :3, :3], 'translations': Hz2[..., :3, -1], 'types':'se3'}

    out2 = multihead_attention(x, z=z, poses_x=poses_x, poses_z=poses_z)

    print('Is it invariant? ',torch.allclose(out, out2, atol=1e-5))
    print((out - out2).abs().max())


if __name__ == "__main__":
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

    import time
    print('Testing Point Invariant Attention v2')
    point_invariant_attention_v_2_test()


    print('Testing Pose Invariant Attention')
    pose_invariant_attention_test()

    print('Testing Point Invariant Attention')

    point_invariant_attention_test()
    
    

