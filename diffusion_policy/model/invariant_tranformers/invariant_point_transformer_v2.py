import copy
from typing import Optional, Any, Union, Callable
from torch import Tensor

import torch
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, MultiheadAttention
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPoseAttention, InvariantPointAttention


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
                 point_dim=4, eps=1e-8, **kwargs):

        super().__init__()
        if kv_dim is None:
            kv_dim = dim

        self.eps = eps
        self.heads = heads

        # num attention contributions
        num_attn_logits = 2

        # qkv projection for scalar attention (normal)
        self.scalar_attn_logits_scale = (num_attn_logits * dim_head) ** -0.5

        self.to_scalar_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_scalar_k = nn.Linear(kv_dim, dim_head * heads, bias=False)
        self.to_scalar_v = nn.Linear(kv_dim, dim_head * heads, bias=False)

        # qkv projection for point attention (coordinate and orientation aware)
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(kv_dim, point_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(kv_dim, point_dim * heads * 3, bias=False)

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

    def forward(self, x, poses_x, z=None, poses_z=None, mask=None):

        x, b, h, eps = x, x.shape[0], self.heads, self.eps
        if z is None:
            z = x
        if poses_z is None:
            poses_z = poses_x

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(z), self.to_scalar_v(z)

        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(z), self.to_point_v(z)

        # split out heads
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                                           (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h=h, c=3),
                                        (q_point, k_point, v_point))

        ## Extract Poses ##
        rotations_q = repeat(poses_x['rotations'], 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations_q = repeat(poses_x['translations'], 'b n c -> (b h) n () c', h=h)
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
        point_dist = (point_qk_diff ** 2).sum(dim=(-1, -2))
        # point_dist = self.point_distance_encoder(point_dist).squeeze(-1)
        # point_dist = point_dist.mean(dim=-1)

        self.point_qk_diff = point_qk_diff

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () ()', b=b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale)

        # combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_points

        # mask

        if exists(mask):
            # mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask[None, ...], 'b i j -> (b h) i j', h=attn_logits.shape[0])
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim=- 1)

        with disable_tf32(), autocast(enabled=False):
            # disable TF32 for precision

            # aggregate values

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h=h)

            # if require_pairwise_repr:
            #     results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum('b n c r, b n d r -> b n d c', rotations_q.transpose(-1, -2),
                                    results_points - translations_q[:, :, 0, None, :])
            results_points_norm = torch.sqrt(torch.square(results_points).sum(dim=-1) + eps)

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h=h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h=h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h=h)

        results = (results_scalar, results_points, results_points_norm)

        # if require_pairwise_repr:
        #     results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h = h)
        #     results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim=-1)
        return self.to_out(results)

###############################################################################################

class InvariantTransformerDecoder(Module):

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: Optional[Module] = None
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x: Tensor, z: Tensor, poses_x, poses_z,
                x_mask: Optional[Tensor] = None,
                z_mask: Optional[Tensor] = None, x_key_padding_mask: Optional[Tensor] = None,
                z_key_padding_mask: Optional[Tensor] = None, x_is_causal: Optional[bool] = None,
                z_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            x: the sequence to the decoder (required).
            z: the sequence from the last layer of the encoder (required).
            x_mask: the mask for the x sequence (optional).
            z_mask: the mask for the z sequence (optional).
            x_key_padding_mask: the mask for the x keys per batch (optional).
            z_key_padding_mask: the mask for the z keys per batch (optional).
            x_is_causal: If specified, applies a causal mask as ``x mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``x_is_causal`` provides a hint that ``x_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            z_is_causal: If specified, applies a causal mask as
                ``z mask``.
                Default: ``False``.
                Warning:
                ``z_is_causal`` provides a hint that
                ``z_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = x

        seq_len = x.shape[1]#_get_seq_len(x, self.layers[0].self_attn.batch_first)
        #x_is_causal = _detect_is_causal_mask(x_mask, x_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, z, poses_x, poses_z,
                         x_mask=x_mask,
                         z_mask=z_mask,
                         x_key_padding_mask=x_key_padding_mask,
                         z_key_padding_mask=z_key_padding_mask,
                         #x_is_causal=x_is_causal,
                         z_is_causal=z_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoderLayer(Module):

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, point_dim=4) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     bias=bias, **factory_kwargs)
        # self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                          bias=bias, **factory_kwargs)

        self.rel_self_attn = InvariantPointAttention(d_model, heads=nhead, dim_head=d_model//nhead, dropout=dropout, kv_dim=None,
                 point_dim=point_dim, eps=1e-8)
        self.rel_multihead_attn = InvariantPointAttention(d_model, nhead, dim_head=d_model//nhead, dropout=dropout, bias=bias, kv_dim=d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        x: Tensor,
        z: Tensor,
        poses_x,
        poses_z,
        x_mask: Optional[Tensor] = None,
        z_mask: Optional[Tensor] = None,
        x_key_padding_mask: Optional[Tensor] = None,
        z_key_padding_mask: Optional[Tensor] = None,
        x_is_causal: bool = False,
        z_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            x: the sequence to the decoder layer (required).
            z: the sequence from the last layer of the encoder (required).
            x_mask: the mask for the x sequence (optional).
            z_mask: the mask for the z sequence (optional).
            x_key_padding_mask: the mask for the x keys per batch (optional).
            z_key_padding_mask: the mask for the z keys per batch (optional).
            x_is_causal: If specified, applies a causal mask as ``x mask``.
                Default: ``False``.
                Warning:
                ``x_is_causal`` provides a hint that ``x_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            z_is_causal: If specified, applies a causal mask as
                ``z mask``.
                Default: ``False``.
                Warning:
                ``z_is_causal`` provides a hint that
                ``z_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = x
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), poses_x, x_mask, x_key_padding_mask, x_is_causal)
            x = x + self._mha_block(self.norm2(x), z, poses_x, poses_z, z_mask, z_key_padding_mask, z_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, poses_x,  x_mask, x_key_padding_mask, x_is_causal))
            x = self.norm2(x + self._mha_block(x, z, poses_x, poses_z, z_mask, z_key_padding_mask, z_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor, poses_x,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        # x = self.self_attn(x, x, x,
        #                    attn_mask=attn_mask,
        #                    key_padding_mask=key_padding_mask,
        #                    is_causal=is_causal,
        #                    need_weights=False)[0]
        x = self.rel_self_attn(x, poses_x, mask=attn_mask)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, poses_x, poses_z,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        # x = self.multihead_attn(x, mem, mem,
        #                         attn_mask=attn_mask,
        #                         key_padding_mask=key_padding_mask,
        #                         is_causal=is_causal,
        #                         need_weights=False)[0]
        x = self.rel_multihead_attn(x=x, z=mem, poses_x=poses_x, poses_z=poses_z, mask=attn_mask)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal




############# MODEL TEST ##############
def geometric_relative_transformer_layer_test():

    import time

    dim = 512
    context_dim = 512
    depth = 5
    heads = 8
    dim_head = 60

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerDecoderLayer(d_model=context_dim, nhead=heads, batch_first=True,
                                    dropout=0.).to(device)

    # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    # z = torch.rand(10, 32, 512)
    # x = torch.rand(20, 32, 512)
    # out = decoder_layer(x, z)
    # 
    # model = InvariantPointTransformer(dim, depth, heads, dim_head, kv_dim=context_dim,
    #                     dropout=0., attention_module=InvariantPointAttention).to(device)

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


def geometry_relative_transformer_test():
    import time

    dim = 512
    context_dim = 512
    depth = 5
    heads = 8
    dim_head = 60

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer = TransformerDecoderLayer(d_model=context_dim, nhead=heads, batch_first=True,
                                    dropout=0.).to(device)


    model = InvariantTransformerDecoder(layer, num_layers=depth).to(device)
    # decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    # z = torch.rand(10, 32, 512)
    # x = torch.rand(20, 32, 512)
    # out = decoder_layer(x, z)
    #
    # model = InvariantPointTransformer(dim, depth, heads, dim_head, kv_dim=context_dim,
    #                     dropout=0., attention_module=InvariantPointAttention).to(device)

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

    geometric_relative_transformer_layer_test()

    geometry_relative_transformer_test()