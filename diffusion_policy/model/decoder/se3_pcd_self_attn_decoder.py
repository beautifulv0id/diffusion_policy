import torch
import torch.nn as nn
from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformerEncoder, URSATransformer
from diffusion_policy.common.pytorch_util import print_dict
class SE3PCDSelfAttnDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.cross_attn1 = URSATransformer(d_model=embedding_dim, nhead=8, num_layers=4)
        self.self_attn = URSATransformerEncoder(d_model=embedding_dim, nhead=8, num_layers=4)
        self.cross_attn2 = URSATransformer(d_model=embedding_dim, nhead=8, num_layers=4)

    def forward(self, tgt, cross_memory, self_memory, query_geometric_args, cross_geometric_args, self_geometric_args):
        nact = tgt.size(1)
        geometric_args = {'query': query_geometric_args, 'key': cross_geometric_args}
        print("CROSS1")
        print("tgt", tgt.shape)
        print("cross_memory", cross_memory.shape)
        print_dict(geometric_args)
        out = self.cross_attn1(tgt, cross_memory, geometric_args=geometric_args)

        out = torch.cat([out, self_memory], dim=1)
        centers = torch.cat([query_geometric_args['centers'], self_geometric_args['centers']], dim=1)
        vectors = torch.cat([query_geometric_args['vectors'], self_geometric_args['vectors']], dim=1)
        geometric_args = {'query': {'centers': centers, 'vectors': vectors}}

        print("SELF")
        print("out", out.shape)
        print_dict(geometric_args)
        self_out = self.self_attn(tgt=out, geometric_args=geometric_args)

        tgt = self_out[:, :nact]
        cross_memory = self_out[:, nact:]
        geometric_args = {'query': {'centers': query_geometric_args['centers'][:, :nact], 'vectors': query_geometric_args['vectors'][:, :nact]},
                          'key': {'centers': geometric_args['query']['centers'][:, nact:], 'vectors': geometric_args['query']['vectors'][:, nact:]}}
        print("CROSS2")
        print("tgt", tgt.shape)  
        print("cross_memory", cross_memory.shape)
        print_dict(geometric_args)
        out = self.cross_attn2(tgt, cross_memory, geometric_args=geometric_args)

        return out