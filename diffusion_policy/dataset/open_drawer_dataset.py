from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_range_normalizer_from_stat
from pathlib import Path
from collections import defaultdict, Counter
import itertools

class OpenDrawerDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['front_rgb', 'front_extrinsic', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', stats: dict = None, **kwargs):
        normalizer = LinearNormalizer()
        if stats is not None:
            for key, stat in stats.items():
                normalizer[key] = get_range_normalizer_from_stat(stat)
        else:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:2]
            }
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        agent_pose = sample['front_extrinsic'][:,2:].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['front_rgb'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pose': agent_pose, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

from rlbench.demo import Demo
from rlbench.utils import get_stored_demos
from diffusion_policy.common.launch_utils import create_dataset
from rlbench.observation_config import ObservationConfig, CameraConfig

class OpenDrawerLowResDataset(BaseImageDataset):
    def __init__(self,
            root, 
            num_episodes=1,
            variation_number=0,
            camera_config_map : Dict[str, CameraConfig] = {
                'left_shoulder': CameraConfig(),
                'right_shoulder': CameraConfig(),
                'overhead': CameraConfig(),
                'wrist': CameraConfig(),
                'front': CameraConfig()
            },
            n_obs=2,
            demo_augmentation_every_n=10,
            use_task_keypoints=True,
            use_pcd = True
            ):
        super().__init__()
        assert variation_number == 0

        root = Path(root)
        num_episodes = num_episodes
        cameras = list(camera_config_map.keys())
        obs_config = ObservationConfig(
            left_shoulder_camera=camera_config_map['left_shoulder'],
            right_shoulder_camera=camera_config_map['right_shoulder'],
            overhead_camera=camera_config_map['overhead'],
            wrist_camera=camera_config_map['wrist'],
            front_camera=camera_config_map['front'],
            gripper_joint_positions=True,
            task_low_dim_state=True,
        )
        
        demos = get_stored_demos(amount = num_episodes,
                                     image_paths = False,
                                     dataset_root = root,
                                     variation_number = variation_number,
                                     task_name = "open_drawer",
                                     obs_config = obs_config,
                                     random_selection = False,
                                     from_episode_number = 0)
        dataset, demo_begin = create_dataset(demos, 
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        n_obs=1, 
                                        use_task_keypoints=use_task_keypoints,
                                        use_pcd=use_pcd)
        self.dataset = dataset
        self.demo_begin = demo_begin
        self.n_obs = n_obs

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_begin = self.demo_begin[idx]
        data = [self.dataset[max(demo_begin, idx-i)] for i in range(self.n_obs)]

        def merge_dicts(dict_list):
            merged_dict = {}
            for k in dict_list[0].keys():
                if isinstance(dict_list[0][k], (np.ndarray, np.float32, int, np.bool_)):
                    merged_dict[k] = np.stack([d[k] for d in dict_list])
                else:
                    merged_dict[k] = merge_dicts([d[k] for d in dict_list])
            return merged_dict

        data = merge_dicts(data)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
def test():
    import os
    # zarr_path = "/home/felix/Workspace/diffusion_policy_felix/data/open_drawer/open_drawer.zarr"
    # print(zarr_path)
    # dataset = OpenDrawerDataset(zarr_path, horizon=16)
    # sample = dataset[0]
    # print(sample['obs']['image'].shape)

    root = "/home/felix/Workspace/diffusion_policy_felix/data/peract/raw/train"
    task_name = "open_drawer"
    variation = 0

    dataset = OpenDrawerLowResDataset(root, num_episodes=1, variation_number=variation, n_obs=3)
    def print_dict(x, indent=0):
        for k in x.keys():
            if isinstance(x[k], dict):
                print(" "*3*indent+k+":")
                print_dict(x[k], indent+1)
            else:
                print(" "*3*indent+k+":", x[k].numpy().shape)
    data = dataset[0]
    print_dict(data)


    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == "__main__":
    test()