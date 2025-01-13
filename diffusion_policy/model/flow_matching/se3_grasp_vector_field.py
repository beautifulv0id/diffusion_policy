import torch
import torch.nn as nn

from geo3dattn.model.common.module_attr_mixin import ModuleAttrMixin


class SE3GraspVectorField(ModuleAttrMixin):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim=64,
                 output_dim=6
                 ):
        super(SE3GraspVectorField, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.out_fn = nn.Linear(latent_dim, output_dim)
        self.grasp_out_fn = nn.Linear(latent_dim, 1)

    def forward(self, x):
        obs_x, obs_f, act_x, act_f = self.encoder(x)
        geo = {'query':act_x, 'key':obs_x}
        out = self.decoder(tgt = act_f, memory = obs_f, geometric_args = geo)
        return self.out_fn(out)

    def encode_obs(self, x):
        obs_x, obs_f = self.encoder.encode_obs(x)
        return obs_x, obs_f

    def encode_act(self, x):
        act_x, act_f = self.encoder.encode_act(x)
        return act_x, act_f

    def set_context(self, obs_x, obs_f, instruction=None):
        self.obs_x = obs_x
        self.obs_f = obs_f
        self.instruction = instruction

    def forward_act(self, x):
        act_x, act_f = self.encoder.encode_act(x['act'])
        time_emb = self.encoder.encode_time(x['time'])
        act_x['time'] = time_emb

        obs_f, act_f = self.encoder.combine_time(self.obs_f, act_f, time_emb)
        if self.instruction is not None:
            act_f = self.encoder.act_combine_instruction(act_f, self.instruction)

        geo = {'query':act_x, 'key':self.obs_x}
        out = self.decoder(tgt = act_f, memory = obs_f, geometric_args = geo)
        return self.out_fn(out), self.grasp_out_fn(out).sigmoid()