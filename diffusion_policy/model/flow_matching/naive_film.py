from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class FILMConditionalBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(out_channels, out_channels)
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.scale = None
        self.bias = None

    def set_context(self, context):
        embed = self.cond_encoder(context)
        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels)
        self.scale = embed[:,0]
        self.bias = embed[:,1]

    
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if self.scale is None:
            self.set_context(cond)
        out = self.scale * out + self.bias

        out = self.blocks[1](out)
        return out
    

class NaiveFilmFlowMatchingModel(nn.Module):
    def __init__(self, 
            in_channels, 
            n_action_steps = 1,
            out_channels = 6, 
            embed_dim = 128,
            mid_dim = 256,
            num_blocks = 4,
            gripper_out = True,
            ignore_collision_out = True,
            ):
        super().__init__()


        self.gripper_out = gripper_out
        self.ignore_collision_out = ignore_collision_out

        self.blocks = nn.ModuleList([
            nn.ModuleList([FILMConditionalBlock(2*embed_dim, mid_dim, embed_dim), nn.ReLU()]),
            *[nn.ModuleList([FILMConditionalBlock(mid_dim, mid_dim, embed_dim), nn.ReLU()]) for _ in range(num_blocks)]])

        out_dim = 6
        if self.gripper_out:
            out_dim += 1
        if self.ignore_collision_out:
            out_dim += 1
        self.to_out = nn.Linear(mid_dim, out_dim)
                
        ## Time Embedings Encoder ##
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim=embed_dim)
        )
        self.embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.SiLU(),
        )    
    
    def get_time_embedding(self, k):
        return self.time_embed(k)
    
    def set_context(self, context):
        obs_features = context['obs_f']
        for (layer, _) in self.blocks:
            layer.set_context(obs_features)

    def forward(self, rot: torch.Tensor, pos: torch.Tensor, k: torch.Tensor, global_cond: torch.Tensor = None):
        H = torch.cat((rot.flatten(1), pos.flatten(1)), dim=-1)
        z = self.embed(H)
        z_time = self.get_time_embedding(k)
        x = torch.cat((z, z_time),dim=-1)
        for (layer, activation) in self.blocks:
            x = layer(x, global_cond)
            x = activation(x)
        out = self.to_out(x)
        out_dict = {
            'v' : x[..., None, :6],
        }
        if self.gripper_out and self.ignore_collision_out:
            out_dict["gripper"] = nn.Sigmoid()(out[..., None, 6])
            out_dict["ignore_collision"] = nn.Sigmoid()(out[..., None, 7])
        if self.gripper_out:
            out_dict["gripper"] = nn.Sigmoid()(out[..., None, 6])
        elif self.ignore_collision_out:
            out_dict["ignore_collision"] = nn.Sigmoid()(out[..., None, 6])
        
        return out_dict
    
