import torch
import torch.nn as nn

import pytorch3d.ops.sample_farthest_points as fps
from pytorch3d.ops.knn import knn_points

from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer


class SuperPointEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=256, n_points_out=14,
                 nheads = 4, num_layers = 2, knn = 10):
        super(SuperPointEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_points_out = n_points_out
        self.knn = knn

        self.linear = nn.Linear(input_dim, output_dim)

        self.model = URSATransformer(output_dim, nhead=nheads, num_layers=num_layers)

    def forward(self, tgt, geometric_args):

        x = tgt
        pcd_x = geometric_args['query']['centers']
        pcd_v = geometric_args['query']['vectors']

        ## Get n points via FPS ##
        out_pcd_x, out_indices = fps(pcd_x, K=self.n_points_out)


        ## Find KNN to Output Points ##
        k = self.knn
        out = knn_points(out_pcd_x, pcd_x, K=k)
        ## Given index (B,Q,K) with B batch and pcd (B,N,3), select (B,Q,K,3) ##
        knn_pcd = torch.gather(pcd_x.unsqueeze(1).expand(-1, self.n_points_out, -1, -1), 2, out[1].unsqueeze(-1).expand(-1, -1, -1, 3))

        ## Prepare data
        ## For the computation we move each SuperPoint to the batch and transform the KNN to the key tokens
        query_f = torch.gather(x, 1, out_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        _query_f = query_f.reshape(-1, query_f.shape[-1])[:,None,:]
        query_c = out_pcd_x
        _query_c = query_c.reshape(-1, query_c.shape[-1])[:,None,:]
        query_v = torch.gather(pcd_v, 1, out_indices[...,None, None].expand(-1, -1, pcd_v.shape[-2], pcd_v.shape[-1]))
        _query_v = query_v.reshape(-1, query_v.shape[-2], query_v.shape[-1])[:,None,...]
        _query_geo = {'centers': _query_c, 'vectors': _query_v}

        key_f = torch.gather(x.unsqueeze(1).expand(-1, self.n_points_out, -1, -1), 2, out[1].unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        _key_f = key_f.reshape(-1, key_f.shape[2], key_f.shape[3])
        key_c = knn_pcd
        _key_c = key_c.reshape(-1, key_c.shape[2], key_c.shape[3])
        key_v = torch.gather(pcd_v.unsqueeze(1).expand(-1, self.n_points_out, -1, -1, -1), 2, out[1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, pcd_v.shape[-2], pcd_v.shape[-1]))
        _key_v = key_v.reshape(-1, key_v.shape[2], key_v.shape[3], key_v.shape[4])
        _key_geo = {'centers': _key_c, 'vectors': _key_v}

        ## Compute the attention
        _geo = {'query': _query_geo, 'key': _key_geo}
        _key_f = self.linear(_key_f)
        _query_f = self.linear(_query_f)
        out_query_f = self.model(_query_f, _key_f, geometric_args=_geo)

        ## Reshape the output
        out_query_f = out_query_f.reshape(-1, self.n_points_out, out_query_f.shape[-1])
        return out_query_f, {'query':{'centers': query_c, 'vectors': query_v}}




if __name__=='__main__':
    # Example usage
    batch = 30
    n_tokens = 3000
    emb_dim = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn((batch, n_tokens, emb_dim)).to(device)

    def random_geometric_args(batch, q_tokens):
        from geo3dattn.common.geometry_random_generator import random_se3_batch
        q_pose = random_se3_batch(batch*q_tokens).reshape(batch, q_tokens, 4, 4)

        return {
            'query': {
                'centers': q_pose[:, :, :3, -1].to(device),
                'vectors': q_pose[:, :, :3, :3].to(device)
            },
        }

    geo_args = random_geometric_args(batch, n_tokens)


    super_encoder = SuperPointEncoder(input_dim=emb_dim, output_dim=emb_dim, n_points_out=200).to(device)
    out_f, out_geo = super_encoder(x, geo_args)
    print(out_f.shape, out_geo['query']['centers'].shape, out_geo['query']['vectors'].shape)
