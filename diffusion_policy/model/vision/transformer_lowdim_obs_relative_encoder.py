from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.common.layer import RelativeCrossAttentionModule
import einops
from scipy.spatial import transform

# TODO: proper naming, remove relative? Change to something w/ SE3?
class TransformerHybridObsRelativeEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
            rotary_embedder: RotaryPositionEncoding3D,
            positional_embedder: SinusoidalPosEmb,
            within_attn : RelativeCrossAttentionModule,
            across_attn : RelativeCrossAttentionModule,
        ):
        """
        Assumes rgb input: B,To,C,H,W
        Assumes low_dim input: B,To,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()

        # handle sharing vision backbone

        obs_shape_meta = shape_meta['obs']
        assert ("agent_pose" in obs_shape_meta), "Must have agent_pose in obs_shape_meta"
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = (n_obs_steps,) + shape
            if type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
    
        self.shape_meta = shape_meta
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.n_obs_steps = n_obs_steps
        self.rotary_embedder = rotary_embedder
        self.query_emb = query_embeddings
        self.keypoint_emb = keypoint_embeddings
        self.re_cross_attn_within = within_attn
        self.re_cross_attn_across = across_attn
        self.positional_embedder = positional_embedder

    def rotary_embed(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, Da)
        Returns:
            torch.Tensor: (B, N, D, 2)
        """
        return self.rotary_embedder(x)
    
    def process_low_dim_features(self, obs_dict):
        batch_size = None
        low_dim_features = list()
        agent_pos : torch.Tensor
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            if key == 'agent_pose':
                agent_pose = data
            elif key == 'keypoint_pcd':
                keypoint_pcd = data
            elif key == 'keypoint_idx':
                keypoint_features = self.keypoint_emb(data.int())
            else:
                low_dim_features.append(data)
        if len(low_dim_features) > 0:
            low_dim_features = torch.cat(low_dim_features, dim=1)
        return agent_pose, keypoint_pcd, keypoint_features, low_dim_features
    
    def forward(self, obs_dict):
        batch_size = obs_dict[self.low_dim_keys[0]].shape[0]
        N = self.n_obs_steps
        
        # process lowdim input
        # (B,N,D), (B,N,D,H,W)
        agent_pose, keypoint_pcd, keypoint_features, low_dim_features = \
            self.process_low_dim_features(obs_dict)
        
        agent_pos = agent_pose[:, :, :3, 3]
        
        # compoute context features
        # (N,B*N,D)
        context_features = einops.rearrange(keypoint_features, 'b n m d -> m (b n) d')
        # (B*N,M,D)
        context_pos = einops.repeat(keypoint_pcd, 'b n m d -> (b n) m d')
        # (B*N,M,D,2)
        context_pos = self.rotary_embed(context_pos)

        # compute query features
        # (1,B*N,D)
        query = einops.repeat(self.query_emb.weight, "n d -> 1 (b n) d", b=batch_size)
        # (B*N,1,D)
        query_pos = einops.repeat(agent_pos, "b n d -> (b n) 1 d")
        # (B*N,1,D,2)
        query_pos = self.rotary_embed(query_pos).type(query.dtype)

        # cross attention within observation
        # (L,B*N,D)
        obs_embs = self.re_cross_attn_within(query=query, value=context_features, 
                                             query_pos=query_pos, value_pos=context_pos)
        # (B*N,D)
        obs_embs = obs_embs[-1].squeeze(0)
        # (N,B,D)
        obs_embs = einops.rearrange(obs_embs, '(b n) d -> n b d', b=batch_size)
        # (N,1,D)
        obs_pos_embs = self.positional_embedder(torch.arange(self.n_obs_steps, dtype=obs_embs.dtype, device=obs_embs.device)).unsqueeze(1)
        # (N,B,D)
        obs_embs = obs_embs + obs_pos_embs

        # cross attention across observations
        obs_emb = self.re_cross_attn_across(query=obs_embs[-1:], 
                                            value=obs_embs[:-1])[0]\
                                                .squeeze(0)

        result = obs_emb
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.n_obs_steps
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,To) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

import hydra
from omegaconf import OmegaConf
import pathlib

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath(
        'config')),
    config_name='train_diffusion_unet_lowdim_relative_workspace.yaml'
)
def test(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    obs_encoder : TransformerHybridObsRelativeEncoder = hydra.utils.instantiate(cfg.policy.obs_encoder)
    out = obs_encoder.output_shape()
    print(out)


if __name__ == "__main__":
    test()