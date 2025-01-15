import numpy as np
import torch
import zarr
from collections import defaultdict, Counter
import itertools
import os
import random
import pickle
from time import time
from pathlib import Path
from diffusion_policy.common.rlbench_util import create_obs_state_plot, convert_rlbench_action, unconvert_rlbench_action
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.so3_util import normal_so3
from diffusion_policy.dataset.rlbench_utils import Resize
from rlbench.backend.const import LOW_DIM_PICKLE

def create_sample_indices(split : zarr.hierarchy.Group, taskvar, n_episodes, n_obs):
    indices = []
    for (task, var) in taskvar:
        taskvar_group = split[task][var]
        for i, demo_group in enumerate(taskvar_group.values()):
            trajectory_length = demo_group['state_action']['proprioception'].shape[0]
            for action_idx in range(1, trajectory_length):
                obs_idxs = []
                for offset in reversed(range(1, n_obs+1)):
                    if action_idx - offset < 0:
                        obs_idxs.append(0)
                    else:
                        obs_idxs.append(action_idx-offset)
                indices.append({
                    'task': task,
                    'var': var,
                    'demo': demo_group,
                    'obs_idxs': obs_idxs,
                    'action_idx': action_idx
                })
    if n_episodes > 0:
        indices = random.sample(indices, n_episodes)
    return indices

def collate_samples(datum, instructions, use_pc, use_rgb, use_mask, apply_cameras, use_lowdim_pcd, use_features):
    sample = {
        'obs': dict(),
        'action': dict()
    }

    obs_idxs = datum['obs_idxs']
    next_keypoint_idx = datum['action_idx']
    cameras = datum['demo']['cameras']
    state_action = datum['demo']['state_action']

    if use_lowdim_pcd:
        sample['obs']['pcd'] = datum['demo']['low_dim_pcd'][obs_idxs[-1]]
    else:
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

    task = datum['task']
    var = datum['var']
    # Sample one instruction feature
    if instructions:
        instr = random.choice(instructions[task][var])
        instr = instr[None].repeat(1, 1, 1)
    else:
        instr = torch.zeros((1, 53, 512))

    sample['obs']['instr'] = instr
    sample['obs']['task'] = task

    return sample

def collate_samples_fused(datum, feature_res):
    sample = {
        'obs': dict(),
        'action': dict()
    }
    obs_idxs = datum['obs_idxs']
    next_keypoint_idx = datum['action_idx']
    obs_group = datum['demo']['fused_cameras'][feature_res]
    state_action = datum['demo']['state_action']

    sample['obs']['pcd'] = obs_group['pcd'][obs_idxs[-1]]
    sample['obs']['clip_features'] = obs_group['clip_features'][obs_idxs[-1]]

    curr_gripper = state_action['proprioception'][obs_idxs]
    sample['obs']['curr_gripper'] = curr_gripper
    sample['obs']['low_dim_state'] = curr_gripper[:,7:8]

    sample['action']['gt_trajectory'] = state_action['proprioception'][next_keypoint_idx].reshape(1, -1)
    return sample
    
def add_noise_to_gripper_pose(gripper_pose, rot_noise_scale, pos_noise_scale):
    if rot_noise_scale == 0 and pos_noise_scale == 0:
        return gripper_pose
    
    t, _ = gripper_pose.shape
    gripper_r, gripper_p, ret = convert_rlbench_action(gripper_pose)

    # add noise to rotation
    if rot_noise_scale > 0:
        drot = normal_so3(t, scale=rot_noise_scale).to(gripper_r.device)
        gripper_r = torch.bmm(drot, gripper_r)

    # add noise to translation
    if pos_noise_scale > 0:
        dpos = torch.normal(0, pos_noise_scale, (t, 3)).to(gripper_pose.device)
        gripper_p += dpos

    gripper_pose = unconvert_rlbench_action(gripper_p, gripper_r, ret)
    return gripper_pose

class RLBenchDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 root: str,
                 instructions = None,
                 cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                 taskvar = [('open_drawer', 0)],
                 use_rgb = True,
                 use_pcd = True,
                 use_mask = True,
                 use_lowdim_pcd = False,
                 use_features = False,
                 rot_noise_scale=0.0,
                 pos_noise_scale=0.0,
                 n_obs_steps = 3,
                 n_episodes = 1,
                 image_rescale=(1.0, 1.0),
                 cache_size=0,
                 split='train',
                 use_precomputed_features=False,
                 feature_res=None
                 ):
        
        self._training = True if split == 'train' else False

        if self._training:
            self._resize = Resize(scales=image_rescale)

        root = Path(root)
        split_path = root / split

        print(f"Loading dataset from {split_path}")
        print("Cache size: ", cache_size)

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        this_taskvar = []
        for root_, (task, var) in itertools.product([split_path], taskvar):
            data_dir = root_ / task / str(var)
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1
                this_taskvar.append((task, var))


        # read from zarr dataset
        split_root = zarr.open(split_path, 'r')
        indices = create_sample_indices(split_root, this_taskvar, n_episodes, n_obs_steps)

        self.indices = indices
        self.cameras = cameras
        self.use_rgb = use_rgb
        self.use_pcd = use_pcd
        self.use_mask = use_mask
        self.use_lowdim_pcd = use_lowdim_pcd
        self.use_features = use_features
        self._cache = dict()
        self._cache_size = cache_size
        self.rot_noise_scale = rot_noise_scale
        self.pos_noise_scale = pos_noise_scale
        self.split = split
        self._root = root
        self.taskvar = taskvar
        self.n_obs_steps = n_obs_steps
        self.n_episodes = n_episodes
        self.image_rescale = image_rescale
        self.cache_size = cache_size
        self.use_precomputed_features = use_precomputed_features
        self.feature_res = feature_res

        print(f"Loaded {len(self)} {split} samples")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):      
        if idx in self._cache:
            sample = self._cache[idx]
        else:
            index = self.indices[idx]
            if self.use_precomputed_features:
                assert self.feature_res is not None
                sample = collate_samples_fused(index, self.feature_res)
            else:
                sample = collate_samples(
                    index,
                    self._instructions,
                    use_pc=self.use_pcd,
                    use_rgb=self.use_rgb,
                    use_mask=self.use_mask,
                    apply_cameras=self.cameras,
                    use_lowdim_pcd=self.use_lowdim_pcd,
                    use_features=self.use_features
                )

            sample = dict_apply(sample, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)

            if len(self._cache) == self._cache_size and self._cache_size > 0:
                key = list(self._cache.keys())[int(time()) % self._cache_size]
                del self._cache[key]

            if len(self._cache) < self._cache_size:
                self._cache[idx] = sample

        if self._training:
            if not self.use_precomputed_features and not self.use_lowdim_pcd:
                sample['obs'].update(self._resize(rgb=sample['obs']['rgb'], pcd=sample['obs']['pcd'], mask=sample['obs'].get('mask', None)))
            sample['obs']['curr_gripper'] = add_noise_to_gripper_pose(sample['obs']['curr_gripper'], self.rot_noise_scale, self.pos_noise_scale)

        return sample
    
    def get_dataset(self, split):
        dataset = RLBenchDataset(
            root=self._root,
            cameras=self.cameras,
            taskvar=self.taskvar,
            use_rgb=self.use_rgb,
            use_pcd=self.use_pcd,
            use_mask=self.use_mask,
            n_obs_steps=self.n_obs_steps,
            n_episodes=self.n_episodes,
            image_rescale=self.image_rescale,
            cache_size=self.cache_size,
            split=split,
            use_precomputed_features=self.use_precomputed_features,
            use_lowdim_pcd=self.use_lowdim_pcd,
            feature_res=self.feature_res
        )
        return dataset

    def get_test_dataset(self):
        dataset = self.get_dataset('test')
        dataset._training = False
        return dataset

    def get_validation_dataset(self):
        dataset = self.get_dataset('val')
        dataset._training = False
        return dataset
    
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
    
    def get_stats(self, relative_to_gripper=False, quaternion_format='xyzw'):
        from pytorch3d.transforms import quaternion_to_matrix
        act_stats = []
        for idx in range(len(self)):
            index = self.indices[idx]
            sample = collate_samples(
                index,
                self._instructions,
                use_pc=False,
                use_rgb=False,
                use_mask=False,
                apply_cameras=False,
                use_lowdim_pcd=False,
                use_features=False
            )

            act_p = torch.from_numpy(sample['action']['gt_trajectory'][..., :3])
            if relative_to_gripper:
                rel_to = torch.from_numpy(sample['obs']['curr_gripper'][-1])
                # The following code expects wxyz quaternion format!
                if quaternion_format == 'xyzw':
                    rel_to[..., 3:7] = rel_to[..., (6, 3, 4, 5)]
                    H_rel_to = quaternion_to_matrix(rel_to[..., 3:7])
                    H_rel_to_inv = H_rel_to.inverse()
                    act_p = torch.einsum('ij,nj->ni', H_rel_to_inv, act_p - rel_to[..., :3])
            act_stats.append(act_p)

        act_stats = torch.cat(act_stats, dim=0)
        return act_stats

    def get_mean_std(self, relative_to_gripper=False, quaternion_format='xyzw'):
        act_stats = self.get_stats(relative_to_gripper, quaternion_format)
        act_p_mean = act_stats.mean(dim=0, keepdim=True)
        act_p_std = act_stats.std(dim=0, keepdim=True)

        act_r_mean = torch.zeros_like(act_p_mean)
        act_r_std = torch.ones_like(act_p_std) * torch.pi

        act_mean = torch.cat([act_p_mean, act_r_mean], dim=-1)
        act_std = torch.cat([act_p_std, act_r_std], dim=-1)
        return act_mean, act_std
    
# TEST CODE
def plot(pcd, rgb=None, batch_idx=0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if rgb is None:
            ax.scatter(pcd[batch_idx,:,0], pcd[batch_idx,:,1], pcd[batch_idx,:,2])
        else:
            ax.scatter(pcd[batch_idx,:,0], pcd[batch_idx,:,1], pcd[batch_idx,:,2], c=rgb[batch_idx])

def extract_rgb_pcd(batch):
    pcd = batch['obs']['pcd']
    pcd = einops.rearrange(pcd, 'n v c h w -> n (v h w) c')
    rgb = batch['obs']['rgb']
    rgb = einops.rearrange(rgb, 'n v c h w -> n (v h w) c')
    return pcd, rgb

def test_dataset():
    from torch.nn import functional as F
    from diffusion_policy.model.common.workspace_cropping import crop_to_workspace
    from diffusion_policy.common.pytorch_util import print_dict

    dataset = RLBenchDataset(
        root=os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/multi_task_test.zarr'),
        cameras=['left_shoulder', 'right_shoulder', 'wrist', 'front'],
        taskvar=[('put_item_in_drawer', 0), ('open_drawer', 0)],
        use_rgb=True,
        use_pcd=True,
        use_mask=False,
        use_lowdim_pcd=False,
        use_features=False,
        n_obs_steps=3,
        n_episodes=-1,
        image_rescale=(1.0, 1.0),
        cache_size=0,
        use_precomputed_features=False
    )

    mean, std = dataset.get_mean_std(relative_to_gripper=True)
    print("Mean: ", mean)
    print("Std: ", std)

    tasks_location_bounds_path = os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_workspace_bounds.json'
    buffer=0.0
    workspace_bounds = get_workspace_bounds(tasks_location_bounds_path, buffer)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(data_loader))

    print_dict(batch)

    print(batch['obs']['pcd'].shape)
    pcd, rgb = extract_rgb_pcd(batch)
    # plot(pcd, rgb)

    batch = next(iter(data_loader))
    pcd = batch['obs']['pcd']
    rgb = batch['obs']['rgb']

    b, v, c, h, w = rgb.shape
    pcd = pcd.reshape(b*v, c, h, w)
    rgb = rgb.reshape(b*v, c, h, w)

    rgb = F.interpolate(rgb, (64, 64), mode='bilinear', align_corners=False)
    pcd = F.interpolate(pcd, (64, 64), mode='nearest')

    batch['obs']['rgb'] = rgb.reshape(b, v, c, 64, 64)
    batch['obs']['pcd'] = pcd.reshape(b, v, c, 64, 64)

    pcd, rgb = extract_rgb_pcd(batch)
    plot(pcd, rgb)


    npts = pcd.shape[1]
    print("Number of points: ", npts)

    cropped_pcd, cropped_rgb = crop_to_workspace(pcd, rgb, workspace_bounds)

    print("Number of points: ", cropped_pcd.shape[1])
    print("Factor of reduction: ", cropped_pcd.shape[1] / pcd.shape[1])
    plot(cropped_pcd, cropped_rgb)


    npts = pcd.shape[0]


    # Farthest point sampling
    tgt_pts = 1000
    ch = pcd.shape[-1]
    # Sample features
    cropped_pcd, out_indices = fps(cropped_pcd, K=tgt_pts)
    cropped_rgb = torch.gather(cropped_rgb, 1, out_indices.unsqueeze(-1).expand(-1, -1, cropped_rgb.shape[-1]))

    plot(cropped_pcd, cropped_rgb)
    plt.show()

    # pcd = pcd[0]
    # rgb = rgb[0]

    # print("FPS PCD shape: ", pcd.shape)
    # print("Factor of reduction: ", npts / pcd.shape[0])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c=rgb)
    # plt.show()


def test_precomputed():
    dataset = RLBenchDataset(
        dataset_path=os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/rlbench.zarr'),
        cameras=['left_shoulder', 'right_shoulder', 'wrist', 'front'],
        task_name='open_drawer',
        use_rgb=True,
        use_pcd=True,
        use_mask=False,
        use_features=False,
        n_obs_steps=3,
        n_episodes=-1,
        image_rescale=(1.0, 1.0),
        cache_size=0,
        use_precomputed_features=True,
        feature_res="res2"
    )

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(data_loader))
    
    plot(batch['obs']['pcd'])
    plt.show()

if __name__ == "__main__":
    import os
    from diffusion_policy.common.pytorch_util import print_dict
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import einops
    from diffusion_policy.common.rlbench_util import get_workspace_bounds
    import pytorch3d.ops.sample_farthest_points as fps

    plt.switch_backend('tkagg')

    # test_precomputed()
    test_dataset()

