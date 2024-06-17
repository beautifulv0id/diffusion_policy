from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from diffusion_policy.model.common.layers import FFWRelativeSelfAttentionModule, FFWRelativeCrossAttentionModule
import einops
from scipy.spatial import transform

class LowDimPointCloudTransformer(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            n_obs_steps: int,
            query_embeddings: nn.Embedding,
            keypoint_embeddings: nn.Embedding,
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
        self.query_emb = query_embeddings
        self.keypoint_emb = keypoint_embeddings
    
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
            elif key == 'low_dim_pcd':
                low_dim_pcd = data
            else:
                low_dim_features.append(data)
        keypoint_features = self.keypoint_emb(torch.arange(low_dim_pcd.shape[-2]).to(self.device))
        keypoint_features = einops.repeat(keypoint_features, 'n d -> b n d', b=batch_size)
        if len(low_dim_features) > 0:
            low_dim_features = torch.cat(low_dim_features, dim=1)
        return agent_pose, low_dim_pcd, keypoint_features, low_dim_features
    
    def forward(self, obs_dict):    
        gripper, low_dim_pcd, keypoint_features, _ = \
            self.process_low_dim_features(obs_dict)
        gripper = torch.cat([gripper[:,-1:], gripper[:,:-1]], dim=1)
        B = gripper.shape[0]
        gripper_features = einops.repeat(self.query_emb.weight, "n d -> b n d", b=B).to(self.device)
        features = torch.cat([gripper_features, keypoint_features], dim=1)
        positions = torch.cat([gripper[:,:,:3,3], low_dim_pcd], dim=1)
        poses = torch.cat([gripper[:,:,:3,:3], torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, low_dim_pcd.shape[1], 1, 1)], dim=1)

        ret = {
            'features': features,
            'positions': positions,
            'poses': poses,
        }
        return ret
    
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
        output_shape = example_output['features'].shape[1:]
        return output_shape

def test():    
    shape_meta = {
        "obs": {
            "agent_pose": {
                "shape": [4, 4]
            },
            "low_dim_pcd": {
                "shape": [10, 3]
            },}
    }
    encoder = LowDimPointCloudTransformer(
        shape_meta=shape_meta,
        n_obs_steps=1,
        query_embeddings=nn.Embedding(1, 60),
        keypoint_embeddings=nn.Embedding(10, 60),
    )
    out = encoder.output_shape()
    print(out)


if __name__ == "__main__":
    test()