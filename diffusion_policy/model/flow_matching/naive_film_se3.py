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
    
    def forward(self, x: torch.Tensor, cond: dict = None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels)
        scale = embed[:,0]
        bias = embed[:,1]
        out = scale * out + bias

        out = self.blocks[1](out)
        return out
    

class NaiveFilmSE3FlowMatchingModel(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels = 6, 
            embed_dim = 128,
            cond_dim = 60):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([FILMConditionalBlock(2*embed_dim, 128, cond_dim), nn.ReLU()]),
            nn.ModuleList([FILMConditionalBlock(128, 256, cond_dim), nn.ReLU()]),
            nn.ModuleList([FILMConditionalBlock(256, out_channels, cond_dim), nn.Identity()])])
        
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

    def forward(self, H: torch.Tensor, k: torch.Tensor, global_cond: torch.Tensor = None):
        H = H.flatten(1)
        z = self.embed(H)
        z_time = self.get_time_embedding(k)
        v = torch.cat((z, z_time),dim=-1)
        for (layer, activation) in self.blocks:
            v = layer(v, global_cond)
            v = activation(v)
        return v
    
