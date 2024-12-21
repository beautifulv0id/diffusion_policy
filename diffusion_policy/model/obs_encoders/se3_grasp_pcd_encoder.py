import torch
import torch.nn as nn
from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin

from geo3dattn.encoder.superpoint_encoder.super_point_encoder import SuperPointEncoder

from geo3dattn.encoder.common.position_encoder import PositionalEncoding


class SE3GraspPointCloudSuperEncoder(ModuleAttrMixin):
    def __init__(self, dim_features=128, depth=3, nheads=4, n_steps_inf=50, n_points_out=100, nhist=3, dim_pcd_features=64):
        super(SE3GraspPointCloudSuperEncoder, self).__init__()

        ## Learnable observation features (Data in Acronym is purely geometrical, no semantics involved)
        self.gripper_features = nn.Parameter(torch.randn(nhist,dim_features))

        ## Learnable action features
        self.action_features = nn.Parameter(torch.randn(1, dim_features))

        ## Pointcloud Encoder ##
        self.obs_encoder = SuperPointEncoder(input_dim=dim_pcd_features, output_dim=dim_features, n_points_out=n_points_out,
                                             nheads=nheads, num_layers=depth)
        
        ## Time Encoder ##
        self.time_encoder = nn.Sequential(
            PositionalEncoding(n_positions=dim_features, max_len=n_steps_inf),
               nn.Linear(dim_features * 2, 4 * dim_features),
                nn.LayerNorm(4 * dim_features),
                nn.GELU(),
                nn.Linear(4 * dim_features, dim_features)
            )

        self.obs_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )

        self.act_merger = nn.Sequential(
                nn.Linear(2 * dim_features, dim_features),
                nn.LayerNorm(dim_features),
                nn.GELU()
            )

    def forward(self, x):
        obs_points, obs_features = self.encode_obs(x['obs'])
        act_points, act_features = self.encode_act(x['act'])
        time_emb = self.encode_time(x['time'])
        obs_f, act_f = self.combine_time(obs_features, act_features, time_emb)
        return obs_points, obs_f, act_points, act_f

    def encode_time(self, time):
        time_emb = self.time_encoder(time)
        return time_emb[:,None,:]

    def encode_obs(self, obs):
        pcd = obs['pcd']
        pcd_features = obs['pcd_features']
        current_gripper = obs['current_gripper']

        batch = pcd.shape[0]
        device = pcd.device

        # encode pcd
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        obs_points = {'centers': pcd, 'vectors': vectors}

        obs_features = pcd_features
        if self.obs_encoder is not None:
            obs_features, obs_geo = self.obs_encoder(tgt=obs_features, geometric_args={'query':obs_points})
            obs_points = obs_geo['query']
        
        # add gripper features
        gripper_features = self.gripper_features[None,...].repeat(batch, 1, 1)
        obs_features = torch.cat((obs_features, gripper_features), dim=1)
        
        vectors = torch.cat((obs_points["vectors"], current_gripper[:,:,:3,:3]), dim=1)
        centers = torch.cat((obs_points["centers"], current_gripper[:,:,:3,-1]), dim=1)
        obs_points["vectors"] = vectors
        obs_points["centers"] = centers

        return obs_points, obs_features

    def encode_act(self, act):
        act_points = {'centers': act[..., :3, -1], 'vectors': act[..., :3, :3]}
        act_features = self.action_features[None,...].repeat(act.shape[0], 1, 1)

        return act_points, act_features

    def combine_time(self, obs_f, act_f, time_emb):
        obs_time_f = self.obs_combine_time(obs_f, time_emb)
        act_time_f = self.act_combine_time(act_f, time_emb)

        return obs_time_f, act_time_f

    def obs_combine_time(self, obs_f, time_emb):
        obs_time_f = torch.cat((obs_f, time_emb.repeat(1, obs_f.shape[1],1)), dim=-1)
        return self.obs_merger(obs_time_f)

    def act_combine_time(self, act_f, time_emb):
        act_time_f = torch.cat((act_f, time_emb.repeat(1, act_f.shape[1],1)), dim=-1)
        return self.act_merger(act_time_f)

class SE3GraspFPSEncoder(SE3GraspPointCloudSuperEncoder):
    def __init__(self, dim_features=128, depth=3, nheads=4, n_steps_inf=50, n_points_out=100, nhist=3, dim_pcd_features=64):
        super(SE3GraspFPSEncoder, self).__init__(dim_features, depth, nheads, n_steps_inf, n_points_out, nhist, dim_pcd_features)
        input_dim = dim_pcd_features
        output_dim = dim_features
        self.linear = nn.Linear(input_dim, output_dim)

    def encode_obs(self, obs):
        obs_pcd_x, obs_pcd_f = super().encode_obs(obs)
        pcd, obs_f = obs['pcd'], obs['pcd_features']
        batch = pcd.shape[0]
        device = pcd.device
        vectors = torch.zeros((3,3))[None,None,:,:].repeat(batch, pcd.shape[1], 1, 1).to(device)
        obs_x = {'centers': pcd, 'vectors': vectors}
        obs_f = self.linear(obs_f)
        return obs_x, obs_f, obs_pcd_x, obs_pcd_f


if __name__=='__main__':
    # Example usage
    batch = 30
    n_tokens = 3000
    emb_dim = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = {
        'obs': {
            'pcd': torch.randn((batch, n_tokens, 3)).to(device),
            'pcd_features': torch.randn((batch, n_tokens, emb_dim)).to(device),
            'current_gripper': torch.randn((batch, 1, 4, 4)).to(device)
        },
        'act': torch.randn((batch, 1, 4, 4)).to(device),
        'time': torch.randn((batch)).to(device)
    }

    encoder = SE3GraspPointCloudSuperEncoder(dim_features=emb_dim, depth=3, nheads=4, n_steps_inf=50, n_points_out=20, nhist=3).to(device)
    obs_points, obs_f, act_points, act_f = encoder(x)
    print("Obs Points:", obs_points['centers'].shape, obs_points['vectors'].shape)
    print("Obs Features:", obs_f.shape)
    print("Act Points:", act_points['centers'].shape, act_points['vectors'].shape)
    print("Act Features:", act_f.shape)