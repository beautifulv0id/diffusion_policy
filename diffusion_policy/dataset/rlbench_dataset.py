import numpy as np
import torch
import zarr
import copy
import os
import pickle
from time import time
from diffusion_policy.common.rlbench_util import create_obs_state_plot, gripper_to_se3, se3_to_gripper
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.so3_util import normal_so3
from diffusion_policy.dataset.rlbench_utils import Resize
from rlbench.backend.const import LOW_DIM_PICKLE

def create_sample_indices(task_group : zarr.hierarchy.Group, n_episodes, n_obs):
    indices = []
    for i, demo_group in enumerate(task_group.values()):
        if i >= n_episodes:
            break
        trajectory_length = demo_group['state_action']['proprioception'].shape[0]
        for action_idx in range(1, trajectory_length):
            obs_idxs = []
            for offset in reversed(range(1, n_obs+1)):
                if action_idx - offset < 0:
                    obs_idxs.append(0)
                else:
                    obs_idxs.append(action_idx-offset)
            indices.append({
                'demo': demo_group,
                'obs_idxs': obs_idxs,
                'action_idx': action_idx
            })
    return indices

class PeractDemoConfig:
    def __init__(self,
                use_pcd=True,
                use_rgb=True,
                use_mask=False,
                low_dim_state=True,
                clip_features=True,
                dino_features=False,
                cameras=['left_shoulder', 'right_shoulder', 'wrist', 'front', 'overhead']):
        self.use_pcd = use_pcd
        self.use_rgb = use_rgb
        self.use_mask = use_mask
        self.low_dim_state = low_dim_state
        self.clip_features = clip_features
        self.dino_features = dino_features
        self.cameras = cameras

def extract_demo_data(demo, idxs, config : PeractDemoConfig):
    data = {}
    for camera in config.cameras:
        data[camera] = {}
        if config.use_pcd:
            data[camera]['pcd'] = demo['cameras'][camera]['pcd'][idxs]
        if config.use_rgb:
            data[camera]['rgb'] = demo['cameras'][camera]['rgb'][idxs]
        if config.use_mask:
            data[camera]['mask'] = demo['cameras'][camera]['mask'][idxs]
        if config.clip_features:
            data[camera]['clip_features']['res1'] = demo['cameras'][camera]['clip_features']['res1'][idxs]
            data[camera]['clip_features']['res2'] = demo['cameras'][camera]['clip_features']['res2'][idxs]
        if config.dino_features:
            data[camera]['dino_features'] = demo['cameras'][camera]['features']['dino_features'][idxs]
    if config.low_dim_state:
        data['low_dim_state'] = demo['state_action']['proprioception'][idxs]
    return data


def collate_samples(datum, use_pc, use_rgb, use_mask, apply_cameras, use_features):
    sample = {
        'obs': dict(),
        'action': dict()
    }
    obs_idxs = datum['obs_idxs']
    next_keypoint_idx = datum['action_idx']
    cameras = datum['demo']['cameras']
    state_action = datum['demo']['state_action']
    if use_pc:
        sample['obs']['pcd'] = np.stack([cameras[camera]['pcd'][obs_idxs[-1]] for camera in apply_cameras])
    if use_rgb:
        rgb = np.stack([cameras[camera]['rgb'][obs_idxs[-1]] for camera in apply_cameras])
        rgb = rgb.astype(np.float32) / 255.0
        sample['obs']['rgb'] = rgb
    if use_mask:
        sample['obs']['mask'] = np.stack([cameras[camera]['mask'][obs_idxs[-1]] for camera in apply_cameras])
    if use_features:
        sample['obs']['clip_features'] = {}
        sample['obs']['clip_features']['res1'] = np.stack([cameras[camera]['clip_features']['res1'][obs_idxs[-1]] for camera in apply_cameras])
        sample['obs']['clip_features']['res2'] = np.stack([cameras[camera]['clip_features']['res2'][obs_idxs[-1]] for camera in apply_cameras])

    curr_gripper = state_action['proprioception'][obs_idxs]
    sample['obs']['curr_gripper'] = curr_gripper
    sample['obs']['low_dim_state'] = curr_gripper[:,7:8]

    sample['action']['gt_trajectory'] = state_action['proprioception'][next_keypoint_idx].reshape(1, -1)
    return sample
    
def add_noise_to_gripper_pose(gripper_pose, rot_noise_scale, pos_noise_scale):
    if rot_noise_scale == 0 and pos_noise_scale == 0:
        return gripper_pose
    
    t, _ = gripper_pose.shape
    gripper_pose, ret = gripper_to_se3(gripper_pose)

    # add noise to rotation
    if rot_noise_scale > 0:
        drot = normal_so3(t, scale=rot_noise_scale).to(gripper_pose.device)
        gripper_pose[:,:3,:3] = torch.bmm(drot, gripper_pose[:,:3,:3])

    # add noise to translation
    if pos_noise_scale > 0:
        dpos = torch.normal(0, pos_noise_scale, (t, 3)).to(gripper_pose.device)
        gripper_pose[:, :3, 3] += dpos

    gripper_pose = se3_to_gripper(gripper_pose, ret)
    return gripper_pose

def load_demos(task_path):
    demos = []
    for demo in os.listdir(task_path):
        if not demo.startswith('demo'):
            continue
        with open(os.path.join(task_path, demo, LOW_DIM_PICKLE), 'rb') as f:
            demo = pickle.load(f)
        demos.append(demo)
    return demos

class RLBenchDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_path: str,
                 cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                 task_name = 'open_drawer',
                 use_rgb = True,
                 use_pcd = True,
                 use_mask = True,
                 use_features = False,
                 rot_noise_scale=0.0,
                 pos_noise_scale=0.0,
                 n_obs_steps = 3,
                 n_episodes = 1,
                 image_rescale=(1.0, 1.0),
                 cache_size=0,
                 split='train',
                 ):
        
        self._training = True

        print(f"Loading dataset from {dataset_path} for task {task_name}")
        print("Cache size: ", cache_size)

        if self._training:
            self._resize = Resize(scales=image_rescale)
        
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        task_group = dataset_root[split][task_name]
        demos = load_demos(os.path.join(dataset_path, split, task_name))
        indices = create_sample_indices(task_group, n_episodes, n_obs_steps)

        self.indices = indices
        self.cameras = cameras
        self.use_rgb = use_rgb
        self.use_pcd = use_pcd
        self.use_mask = use_mask
        self.use_features = use_features
        self.demos = demos
        self._cache = dict()
        self._cache_size = cache_size
        self.rot_noise_scale = rot_noise_scale
        self.pos_noise_scale = pos_noise_scale
        self.split = split
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.n_episodes = n_episodes
        self.image_rescale = image_rescale
        self.cache_size = cache_size

        print(f"Loaded {len(self)} {split} samples")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):      
        if idx in self._cache:
            sample = self._cache[idx]
        else:
            index = self.indices[idx]
            sample = collate_samples(
                index,
                use_pc=self.use_pcd,
                use_rgb=self.use_rgb,
                use_mask=self.use_mask,
                apply_cameras=self.cameras,
                use_features=self.use_features
            )
            sample = dict_apply(sample, lambda x: torch.from_numpy(x))

            if len(self._cache) == self._cache_size and self._cache_size > 0:
                key = list(self._cache.keys())[int(time()) % self._cache_size]
                del self._cache[key]

            if len(self._cache) < self._cache_size:
                self._cache[idx] = sample

        if self._training:
            sample['obs'].update(self._resize(rgb=sample['obs']['rgb'], pcd=sample['obs']['pcd'], mask=sample['obs'].get('mask', None)))
            sample['obs']['curr_gripper'] = add_noise_to_gripper_pose(sample['obs']['curr_gripper'], self.rot_noise_scale, self.pos_noise_scale)

        return sample

    def get_validation_dataset(self):
        val_set = RLBenchDataset(
            dataset_path=self.dataset_path,
            cameras=self.cameras,
            task_name=self.task_name,
            use_rgb=self.use_rgb,
            use_pcd=self.use_pcd,
            use_mask=self.use_mask,
            n_obs_steps=self.n_obs_steps,
            n_episodes=self.n_episodes,
            image_rescale=self.image_rescale,
            cache_size=self.cache_size,
            split='val'
        )
        val_set._training = False
        return val_set
    
    def empty_cache(self):
        for k, v in self._cache.items():
            del v
        self._cache = dict()

    def get_data_visualization(self, num_samples=16):
        if not self.use_rgb:
            return None
        idxs = range(min(num_samples, len(self)))
        imgs = []
        for idx in idxs:
                data = self[idx]
                data = dict_apply(data, lambda x: x.unsqueeze(0))
                img = create_obs_state_plot(data['obs'], gt_action=data['action']['gt_trajectory'], use_mask=False, quaternion_format = 'xyzw')[0]
                imgs.append(torch.from_numpy(img[:3,:,:]))
                if self.use_mask:
                    img = create_obs_state_plot(data['obs'], gt_action=data['action']['gt_trajectory'], use_mask=True, quaternion_format = 'xyzw')[0]
                    imgs.append(torch.from_numpy(img[:3,:,:])) 
        imgs = torch.stack(imgs) / 255.0
        return imgs

if __name__ == "__main__":
    import os
    from diffusion_policy.common.pytorch_util import print_dict
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    dataset = RLBenchDataset(
        dataset_path=os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/peract.zarr'),
        cameras=['left_shoulder', 'right_shoulder', 'wrist', 'front'],
        task_name='open_drawer',
        use_rgb=True,
        use_pcd=True,
        use_mask=False,
        use_features=True,
        n_obs_steps=3,
        n_episodes=1,
        image_rescale=(1.0, 1.0),
        cache_size=0
    )

    sample = dataset[0]

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(data_loader))
    print_dict(batch)
