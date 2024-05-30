from typing import Union, Dict
import torch.nn as nn
import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class SharedPositionNormalizer(ModuleAttrMixin):
    '''
    Applies a scaling and offset to the positions in the data.
    '''

    def __init__(self, offset = 0., scale = 1.):
        super(SharedPositionNormalizer, self).__init__()
        self.scale = scale

    def normalize(self, data):
        assert 'obs' in data
        obs = data['obs']

        for key in obs.keys():
            if key.find('rot') != -1:
                key_pos = key.replace('rot', 'pos')
                pos = obs[key_pos]
                new_pos = pos * self.scale
                obs[key_pos] = torch.tanh(new_pos)

        if 'action' in data:
            action = data['action']
            act_p = action['act_p']
            action['act_p'] = act_p * self.scale
            data['action'] = action

        return data

    def unnormalize(self, data):

        action = data['action']
        act_p = action['act_p']
        action['act_p'] = act_p / self.scale
        data['action'] = action

        return data