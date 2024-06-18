import math

import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


task_n_keypoints = {
    'put_item_in_drawer': 28,
    'open_drawer': 25,
    'stack_blocks': 63,
}

class LowdimKeypointEmbedder(ModuleAttrMixin):
    def __init__(self, n_pcd, n_obs_steps, dim_out=60):
        super(LowdimKeypointEmbedder, self).__init__()

        ## Parameters ##
        self.dim_out = dim_out
        self.pcd_features = nn.Parameter(torch.randn(n_pcd, dim_out))
        self.gripper_features = nn.Parameter(torch.randn(n_obs_steps, dim_out))

    def forward(self, obs):
        pcd = obs['low_dim_pcd'][:,-1]
        embeddings = self.pcd_features.data
        new_obs = {}
        new_obs['context_pcd'] = pcd
        new_obs['context_features'] = embeddings
        new_obs['gripper_pcd'] = obs['robot0_eef_pos']
        new_obs['gripper_rot'] = obs['robot0_eef_rot']
        new_obs['gripper_features'] = self.gripper_features.data
        new_obs['low_dim_state'] = obs['low_dim_state']
        return new_obs

    def get_args(self):
        return {
            '__class__': [type(self).__module__, type(self).__name__]
        }

def test():
    embedder = LowdimKeypointEmbedder(10, 2)
    context = {'low_dim_pcd': torch.randn(64, 1, 10, 3), 'robot0_eef_pos': torch.randn(64, 2, 3)}
    out = embedder(context)
    print("point cloud shape", out['context_pcd'].shape)
    print("embedding shape", out['context_features'].shape)

if __name__ == "__main__":
    test()
    print("Test passed")