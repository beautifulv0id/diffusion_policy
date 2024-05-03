import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer
from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb
from diffusion_policy.model.common.layers import FFWRelativeCrossAttentionModule
import einops

# TODO: proper naming, remove relative? Change to something w/ SE3?
class GITLowdimEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
            positional_embedder: SinusoidalPosEmb,
            across_attn : FFWRelativeCrossAttentionModule,
            embed_dim: int = 60,
            depth: int = 2,
            num_heads: int = 4,
            mlp_dim: int = 256,
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
            key_shape_map[key] = (n_obs_steps,) + shape if key == 'agent_pose' else shape
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
        self.positional_embedder = positional_embedder
        self.keypoint_emb = keypoint_embeddings
        self.query_emb = query_embeddings

        self.re_inv_cross_attn_within = GeometryInvariantTransformer(
            dim=embed_dim,
            depth=depth,
            dim_head=embed_dim,
            heads=num_heads,
            mlp_dim=mlp_dim,
            kv_dim=embed_dim,
            dropout=0.,
            return_last_attmap=False
        )
        self.re_cross_attn_across = across_attn


    
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
            assert data.shape[1:] == self.key_shape_map[key], f"Expected shape {self.key_shape_map[key]} but got {data.shape[1:]} for key {key}"
            if key == 'agent_pose':
                agent_pose = data
            elif key == 'low_dim_pcd':
                low_dim_pcd = data
            else:
                low_dim_features.append(data)
        keypoint_features = self.keypoint_emb(torch.arange(low_dim_pcd.shape[-2]).to(low_dim_pcd.device)).unsqueeze(0).repeat(batch_size, 1, 1)
        if len(low_dim_features) > 0:
            low_dim_features = torch.cat(low_dim_features, dim=1)
        return agent_pose, low_dim_pcd, keypoint_features, low_dim_features
    
    def forward(self, obs_dict):
        batch_size = obs_dict[self.low_dim_keys[0]].shape[0]
        N = self.n_obs_steps
        
        # process lowdim input
        # (B,N,D), (B,H,N,D)
        agent_pose, low_dim_pcd, keypoint_features, low_dim_features = \
            self.process_low_dim_features(obs_dict)
                
        # compoute context features
        # (B,N,D)
        context_features = keypoint_features
        # (B,N,3)
        context_pos = low_dim_pcd

        # compute query features
        # (1,B*N,D)
        query = einops.repeat(self.query_emb.weight, "n d -> b n d", b=batch_size)
        # (B*N,1,D)
        query_pose = einops.repeat(agent_pose, "b n i j -> b n i j")

        # cross attention within observation
        extras = {'x_poses': query_pose, 'z_poses': context_pos,
              'x_types':'se3', 'z_types':'3D'}
        obs_embs = self.re_inv_cross_attn_within(x=query, z=context_features, extras=extras)
        # (N,B,D)
        obs_embs = einops.rearrange(obs_embs, 'b n d -> n b d')
        # (N,1,D)
        obs_pos_embs = self.positional_embedder(torch.arange(self.n_obs_steps, dtype=obs_embs.dtype, device=obs_embs.device)).unsqueeze(1)
        # (N,B,D)
        obs_embs = obs_embs + obs_pos_embs

        # cross attention across observations
        if obs_embs.shape[0] > 1:
            obs_emb = self.re_cross_attn_across(query=obs_embs[-1:], 
                                                value=obs_embs[:-1])[0]\
                                                    .squeeze(0)
        else:
            obs_emb = obs_embs[0]

        result = obs_emb
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        To = self.n_obs_steps
        agent_pose_shape = tuple(obs_shape_meta.pop('agent_pose')['shape'])
        agent_pose = torch.zeros(
            (batch_size,To) + agent_pose_shape, 
            dtype=self.dtype,
            device=self.device)
        example_obs_dict['agent_pose'] = agent_pose
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
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
    config_name='train_flow_matching_unet_lowdim_workspace.yaml',
)
def test(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    obs_encoder : GITLowdimEncoder = hydra.utils.instantiate(cfg.policy.observation_encoder)
    out = obs_encoder.output_shape()
    print(out)


if __name__ == "__main__":
    test()