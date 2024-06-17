import math
import numpy as np
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb


class NaivePolicy(nn.Module):
    def __init__(self, dim=9, output_size=3, context_dim=2, hidden1_size=120, hidden2_size=120, n_context=1):
        super(NaivePolicy, self).__init__()

        self.context_encoder = nn.Linear(n_context*context_dim, hidden1_size)

        self.layer1 = nn.Linear(dim, hidden1_size)
        self.layer2 = nn.Linear(2*hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, hidden2_size)
        self.layer4 = nn.Linear(hidden2_size, output_size)

        self.time_embedding = SinusoidalPosEmb(hidden1_size)

    def forward(self, x, t):

        _x = x.reshape(x.shape[0], -1)

        x = F.relu(self.layer1(_x))

        ## Join Time Embedding with Input ##
        t_emb = self.time_embedding(t)
        x = x + t_emb

        ## Join Context with Position ##
        xz = torch.cat((x, self.z_context), dim=-1)

        x = F.relu(self.layer2(xz))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

    def set_context(self, context):
        self.z_context = self.context_encoder(context.reshape(context.shape[0], -1))
