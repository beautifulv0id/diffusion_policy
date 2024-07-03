import numpy as np
import torch
import zarr
import copy
from time import time
from diffusion_policy.common.rlbench_util import create_obs_state_plot
from diffusion_policy.common.pytorch_util import dict_apply, print_dict
from diffusion_policy.dataset.rlbench_utils import Resize

def create_sample_indices(episode_ends, keypoints, keypoint_ends, n_obs, use_keypoint_obs_only=True, n_skip=1, episode_start=0, keypoint_start=0):
    indices = []
    episode_ends = np.concatenate([[episode_start], episode_ends])
    keypoint_ends = np.concatenate([[keypoint_start], keypoint_ends])
    for ep_idx in range(len(episode_ends)-1):
        episode_keypoints = keypoints[keypoint_ends[ep_idx]:keypoint_ends[ep_idx+1]]
        episode_start = episode_ends[ep_idx]
        episode_end = episode_ends[ep_idx+1]
        next_keypoint_idx = 0
        for obs_idx in range(episode_start, episode_end, n_skip):
            if use_keypoint_obs_only:
                if obs_idx not in episode_keypoints and obs_idx != episode_start:
                    continue
            while obs_idx >= episode_keypoints[next_keypoint_idx] and next_keypoint_idx < len(episode_keypoints) - 1:
                next_keypoint_idx += 1
            if obs_idx >= episode_keypoints[next_keypoint_idx]:
                break
            obs_idxs = []
            for i in reversed(range(1, n_obs)):
                offset = i
                if obs_idx == episode_keypoints[next_keypoint_idx-1]:
                    offset = i + 1
                if next_keypoint_idx - offset < 0:
                    obs_idxs.append(episode_start)
                else:
                    obs_idxs.append(episode_keypoints[next_keypoint_idx-offset])
            obs_idxs.append(obs_idx)
            action_idxs = episode_keypoints[next_keypoint_idx]
            indices.append([obs_idxs, action_idxs])
    return indices

def collate_samples(camera_data, gripper_pose, gripper_open, ignore_collisions, gripper_joint_positions, obs_idxs, next_keypoint_idx, use_pc, use_rgb, use_mask, apply_cameras, use_low_dim_state):
    sample = {
        'obs': dict(),
        'action': dict()
    }
    if use_pc:
        sample['obs']['pcd'] = np.stack([camera_data[f'{camera}_point_cloud'][obs_idxs[-1]] for camera in apply_cameras])
    if use_rgb:
        sample['obs']['rgb'] = np.stack([camera_data[f'{camera}_rgb'][obs_idxs[-1]] for camera in apply_cameras])
    if use_mask:
        sample['obs']['mask'] = np.stack([camera_data[f'{camera}_mask'][obs_idxs[-1]] for camera in apply_cameras])

    sample['obs']['curr_gripper'] = np.stack([np.concatenate([gripper_pose[obs_idx], gripper_open[obs_idx], ignore_collisions[obs_idx]]) for obs_idx in obs_idxs])
    if use_low_dim_state:
        sample['obs']['low_dim_state'] = np.stack([np.concatenate([gripper_open[obs_idx], gripper_joint_positions[obs_idx]]) for obs_idx in obs_idxs])

    sample['action']['gt_trajectory'] = np.concatenate([gripper_pose[next_keypoint_idx], gripper_open[next_keypoint_idx], ignore_collisions[obs_idxs[-1]]]).reshape(1,9)
    return sample

class RLBenchNextBestPoseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                 task_name = 'open_drawer',
                 use_rgb = True,
                 use_pcd = True,
                 use_mask = True,
                 use_low_dim_state = True,
                 use_keypoint_obs_only = True,
                 n_skip = 1,
                 n_obs_steps = 3,
                 n_episodes = 1,
                 val_ratio = 0.1,
                 image_rescale=(1.0, 1.0),
                 cache_size=0
                 ):
        
        self._training = True

        if self._training:
            self._resize = Resize(scales=image_rescale)
        
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        task_root = dataset_root[task_name]
        data_root = task_root['data']

        camera_data = dict()

        for camera in cameras:
            if use_rgb:
                # float32, [0,255] -> [0,1], (N,3,H,W)
                image_data = data_root[f'{camera}_rgb'][:]
                image_data = image_data.astype(np.float32) / 255.0
                camera_data[f'{camera}_rgb'] = image_data
            if use_pcd:
                # float32, (N,3,H,W)
                pcd_data = data_root[f'{camera}_point_cloud'][:]
                camera_data[f'{camera}_point_cloud'] = pcd_data
            if use_mask:
                # float32, [0,1], (N,1,H,W)
                mask_data = data_root[f'{camera}_mask'][:]
                camera_data[f'{camera}_mask'] = mask_data

        # float32, (N,7)
        gripper_pose = data_root['gripper_pose'][:]

        # bool, (N,1)
        gripper_open = data_root['gripper_open'][:]

        # float32, (N,2)
        gripper_joint_positions = data_root['gripper_joint_positions'][:]

        # bool, (N,1)
        ignore_collisions = data_root['ignore_collisions'][:]

        # int32, (N,1)
        episode_ends = task_root['meta']['episode_ends'][:]

        # int32, (N,1)
        keypoints = task_root['meta']['keypoints'][:]

        # int32, (N,1)
        keypoint_ends = task_root['meta']['keypoint_ends'][:]

        # Demo, (N,)
        demos = task_root['meta']['demos'][:]

        if n_episodes == -1:
            n_episodes = len(episode_ends)

        if n_episodes is not None:
            episode_ends = episode_ends[:n_episodes]
            keypoint_ends = keypoint_ends[:n_episodes]
            demos = demos[:n_episodes]

        n_val = min(max(0, round(n_episodes * val_ratio)), n_episodes-1)

        train_indices = create_sample_indices(
            episode_ends=episode_ends[:n_episodes-n_val],
            keypoints=keypoints,
            keypoint_ends=keypoint_ends[:n_episodes-n_val],
            n_obs=n_obs_steps,
            n_skip=n_skip,
            use_keypoint_obs_only=use_keypoint_obs_only
        )
        if n_val > 0:
            val_indices = create_sample_indices(
                episode_ends=episode_ends[n_episodes-n_val:],
                keypoints=keypoints,
                keypoint_ends=keypoint_ends[n_episodes-n_val:],
                n_obs=n_obs_steps,
                n_skip=n_skip,
                use_keypoint_obs_only=use_keypoint_obs_only,
                episode_start=episode_ends[n_episodes-n_val-1],
                keypoint_start=keypoint_ends[n_episodes-n_val-1]
            )
        else:
            val_indices = []

        self.camera_data = camera_data
        self.gripper_pose = gripper_pose
        self.gripper_open = gripper_open
        self.ignore_collisions = ignore_collisions
        self.gripper_joint_positions = gripper_joint_positions
        self.episode_ends = episode_ends
        self.indices = train_indices
        self.val_indices = val_indices
        self.cameras = cameras
        self.use_rgb = use_rgb
        self.use_pcd = use_pcd
        self.use_mask = use_mask
        self.use_low_dim_state = use_low_dim_state
        self.demos = demos[:n_episodes-n_val]
        self.val_demos = demos[n_episodes-n_val:]
        self._cache = dict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        
        indices = self.indices[idx]
        obs_idxs, next_keypoint_idx = indices
        sample = collate_samples(
            camera_data=self.camera_data,
            gripper_pose=self.gripper_pose,
            gripper_open=self.gripper_open,
            ignore_collisions=self.ignore_collisions,
            gripper_joint_positions=self.gripper_joint_positions,
            obs_idxs=obs_idxs,
            next_keypoint_idx=next_keypoint_idx,
            use_pc=self.use_pcd,
            use_rgb=self.use_rgb,
            use_mask=self.use_mask,
            apply_cameras=self.cameras,
            use_low_dim_state=self.use_low_dim_state
        )
        sample = dict_apply(sample, lambda x: torch.tensor(x))

        if self._training:
            sample['obs'].update(self._resize(rgb=sample['obs']['rgb'], pcd=sample['obs']['pcd'], mask=sample['obs'].get('mask', None)))

        if len(self._cache) == self._cache_size and self._cache_size > 0:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[idx] = sample

        return sample

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.val_indices
        val_set.demos = self.val_demos
        return val_set

    def get_data_visualization(self, num_samples=16):
        if not self.use_rgb:
            return None
        idxs = np.random.randint(0, len(self), size=num_samples)
        imgs = []
        for idx in idxs:
                data = self[idx]
                data = dict_apply(data, lambda x: x.unsqueeze(0))
                img = create_obs_state_plot(data['obs'], data['action']['gt_trajectory'], use_mask=False, quaternion_format = 'xyzw')
                imgs.append(torch.from_numpy(img[:3,:,:]))
                if self.use_mask:
                    img = create_obs_state_plot(data['obs'], data['action']['gt_trajectory'], use_mask=True, quaternion_format = 'xyzw')
                    imgs.append(torch.from_numpy(img[:3,:,:])) 
        imgs = torch.stack(imgs) / 255.0
        return imgs


def speed_test():
    import time
    import os
    dataset_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], "data/image.zarr")
    dataset = RLBenchNextBestPoseDataset(dataset_path, n_skip=1, n_episodes=-1, val_ratio=0.1, use_mask=True, image_rescale=(0.7, 1.25), cache_size=100)
    print(len(dataset)) 
    start = time.time()
    for i in range(100000):
        data = dataset[i % len(dataset)]
    print(time.time() - start)


if __name__ == "__main__":
    from PIL import Image
    from torchvision.utils import make_grid, save_image
    speed_test()
    # dataset_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], "data/image.zarr")
    # dataset = RLBenchNextBestPoseDataset(dataset_path, n_skip=1, n_episodes=-1, val_ratio=0.1, use_mask=True, image_rescale=(0.7, 1.25))
    # print(len(dataset)) 
    # print_dict(dataset[0])
    # val_set = dataset.get_validation_dataset()
    # print(len(val_set))
    # print_dict(val_set[0])

    # rgb = dataset[0]['obs']['rgb']
    # img = make_grid(rgb, nrow=3)
    # save_image(img, 'rgb.png')
            
    # img = dataset.get_data_visualization(20)
    # img = make_grid(img, nrow=4)
    # save_image(img, "data_visualization.png")
