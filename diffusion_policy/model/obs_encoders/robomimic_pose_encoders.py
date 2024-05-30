import math

import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


Objects = {
    'put_item_in_drawer': ['robot0_eef', 'kp0', 'kp1', 'kp2', 'kp3'],
    'open_drawer': ['robot0_eef', 'kp0', 'kp1', 'kp2', 'kp3', 'kp4'],
    'stack_blocks': ['robot0_eef', 'kp0', 'kp1', 'kp2', 'kp3', 'kp4', 'kp5', 'kp6', 'kp7', 'kp8'],
}


class BaseObservationEncoder(ModuleAttrMixin):
    def __init__(self, dim_out=50, task='put_item_in_drawer', n_obs_steps=2,
                 additional_inputs=dict(),
                 add_global_pose=False):
        super(BaseObservationEncoder, self).__init__()

        ## Parameters ##
        self.object_names = Objects[task]
        self.num_objects = len(self.object_names)
        self.n_obs_steps = n_obs_steps
        self.object_features = nn.Parameter(torch.randn(self.num_objects, dim_out))


        self.positional_encoding = PositionalEncoding(dim_out,  max_len=n_obs_steps)

        ## Additional features ##
        self.additional_inputs = additional_inputs
        dim_in = 0
        for key in additional_inputs.keys():
            dim_in += additional_inputs[key]
        if dim_in > 0:
            self.additional_enc = nn.Linear(dim_in, dim_out)

        ## Add global pose Token##
        self.add_global_pose = add_global_pose
        if add_global_pose:
            self.global_pose_enc = nn.Parameter(torch.randn(1, dim_out))
            self.global_p = nn.Parameter(torch.zeros(1, 3), requires_grad=False)
            self.global_r = nn.Parameter(torch.eye(3)[None,...], requires_grad=False)

    def forward(self, context):


        ## Precompute additional features ##
        additional_features = torch.Tensor([]).to(context['robot0_eef_pos'].device)
        for key in self.additional_inputs.keys():
            additional_features = torch.cat((additional_features, context[key][:, :self.n_obs_steps, ...]), dim=-1)
        if additional_features.numel()>0:
            additional_features = self.additional_enc(additional_features)

        obs = {}
        obs['obs_p'] = None
        obs['obs_r'] = None
        obs['obs_f'] = None
        for idx, obj_name in enumerate(self.object_names):

            #### Set Object Position and Rotation ####
            obj_pos_name = obj_name+'_pos'
            if obs['obs_p'] is None:
                obs['obs_p'] = context[obj_pos_name][:, :self.n_obs_steps, ...]
            else:
                obs['obs_p'] = torch.cat((obs['obs_p'], context[obj_pos_name][:, :self.n_obs_steps, ...]), dim=1)

            obj_rot_name = obj_name+'_rot'
            if obs['obs_r'] is None:
                obs['obs_r'] = context[obj_rot_name][:, :self.n_obs_steps, ...]
            else:
                obs['obs_r'] = torch.cat((obs['obs_r'], context[obj_rot_name][:, :self.n_obs_steps, ...]), dim=1)

            ## Set Features ##
            features = self.object_features[None,None, idx, :].repeat(context[obj_pos_name].shape[0], context[obj_pos_name].shape[1], 1)
            if additional_features.numel() > 0:
                features += additional_features[:, -features.shape[1]:, :]
            features = self.positional_encoding(features)
            if obs['obs_f'] is None:
                obs['obs_f'] = features
            else:
                obs['obs_f'] = torch.cat((obs['obs_f'], features), dim=1)


        ## Add global pose token##
        if self.add_global_pose:
            B = obs['obs_p'][:,0,:]
            if 'reference_pos' in context:
                obs_p = context['reference_pos']
                obs_r = context['reference_rot']
            else:
                obs_p = self.global_p[None,...].repeat(B.shape[0], 1, 1)
                obs_r = self.global_r[None,...].repeat(B.shape[0], 1, 1, 1)
            obs_f = self.global_pose_enc[None,...].repeat(B.shape[0], 1, 1)

            obs['obs_p'] = torch.cat((obs_p, obs['obs_p']), dim=1)
            obs['obs_r'] = torch.cat((obs_r, obs['obs_r']), dim=1)
            obs['obs_f'] = torch.cat((obs_f, obs['obs_f']), dim=1)

        return obs

    def get_args(self):
        return {
            '__class__': [type(self).__module__, type(self).__name__]
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x