from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from pathlib import Path
from rlbench.utils import get_stored_demos
from diffusion_policy.common.launch_utils import create_dataset
from rlbench.observation_config import ObservationConfig, CameraConfig


class RLBenchDataset():
    def __init__(self,
            root, 
            task_name="open_drawer",
            num_episodes=1,
            variation=0,
            cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
            image_size=(128, 128),
            n_obs=2,
            demo_augmentation_every_n=10,
            obs_augmentation_every_n=10,
            use_low_dim_pcd=False,
            use_pcd = False,
            use_rgb = False,
            use_depth = False,
            use_mask = False,
            use_pose = False,
            keypoints_only=False,
            use_low_dim_state=False,
            val_ratio=0.0,
            ):

        root = Path(root)
        # TODO: change this
        camera_config = CameraConfig(rgb=use_rgb, depth=use_depth, mask=use_mask, point_cloud=use_pcd, image_size=image_size)
        camera_off_config = CameraConfig()
        camera_off_config.set_all(False)
        use_cameras = any([use_rgb, use_depth, use_mask, use_pcd])
        if not use_cameras:
            cameras = []
        all_cameras = set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'])
        unused_cameras = all_cameras - set(cameras)
        camera_config_map = {k: camera_config for k in cameras}
        camera_config_map.update({k: camera_off_config for k in unused_cameras})
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
                                     variation_number = variation,
                                     task_name = task_name,
                                     obs_config = obs_config,
                                     random_selection = False, #TODO: change this
                                     from_episode_number = 0)
        
        num_episodes = len(demos)
        n_val = min(max(0, round(num_episodes * val_ratio)), num_episodes-1)
        
        # val_idxs = np.random.choice(num_episodes, size=n_val, replace=False)
        # train_idxs = np.array([i for i in range(num_episodes) if i not in val_idxs])
        # train_demos = [demos[i] for i in train_idxs]
        # val_demos = [demos[i] for i in val_idxs]   
        train_demos = demos[:num_episodes-n_val]
        val_demos = demos[num_episodes-n_val:] 
        
        train_dataset, train_demo_begin = create_dataset(train_demos, 
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        obs_augmentation_every_n=obs_augmentation_every_n,
                                        n_obs=n_obs, 
                                        use_low_dim_pcd=use_low_dim_pcd,
                                        use_pcd=use_pcd,
                                        use_rgb=use_rgb,
                                        use_pose=use_pose,
                                        use_low_dim_state=use_low_dim_state,
                                        keypoints_only=keypoints_only)
        
        val_dataset, val_demo_begin = create_dataset(val_demos,
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        obs_augmentation_every_n=obs_augmentation_every_n,
                                        n_obs=n_obs, 
                                        use_low_dim_pcd=use_low_dim_pcd,
                                        use_pcd=use_pcd,
                                        use_rgb=use_rgb,
                                        use_pose=use_pose,
                                        use_low_dim_state=use_low_dim_state,
                                        keypoints_only=keypoints_only)
        
        self.demos = train_demos
        self.dataset = train_dataset
        self.demo_begin = train_demo_begin
        self.val_demos = val_demos
        self.val_dataset = val_dataset
        self.val_demo_begin = val_demo_begin
        self.n_obs = n_obs
        self.keypoints_only = keypoints_only

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.dataset[idx]
        this_data = {
            "action": data["action"],
            "obs": dict()
        }
        this_data['obs']['robot0_eef_rot'] = data['obs']['robot0_eef_rot']
        this_data['obs']['robot0_eef_pos'] = data['obs']['robot0_eef_pos']
        if 'low_dim_state' in data['obs']:
            this_data['obs']['low_dim_state'] = data['obs']['low_dim_state']
        for k in data['obs'].keys():
            if not 'robot0_eef' in k and not 'low_dim_state' in k:
                this_data['obs'][k] = data['obs'][k][-1:]
        torch_data = dict_apply(this_data, torch.from_numpy)
        # torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.dataset = self.val_dataset
        val_set.demo_begin = self.val_demo_begin
        val_set.demos = self.val_demos
        return val_set

    def get_normalizer(self):
        pass


class RLBenchLowdimDataset(RLBenchDataset, BaseLowdimDataset):
    def __init__(self,           
                    root, 
                    use_low_dim_pcd=True,
                    use_pose=True,
                    task_name="open_drawer_keypoint",
                    num_episodes=1,
                    variation=0,
                    cameras=[],
                    image_size=None,
                    n_obs=2,
                    demo_augmentation_every_n=10,
                    obs_augmentation_every_n=10,
                    keypoints_only=False,
                    use_low_dim_state=False,
                    val_ratio=0.0,
                    use_pcd=False,
                    use_rgb=False,
                    use_depth=False,
                    use_mask=False):
        super().__init__(root=root,
                        task_name=task_name,
                        num_episodes=num_episodes,
                        variation=variation,
                        cameras=cameras,
                        image_size=image_size,
                        n_obs=n_obs,
                        demo_augmentation_every_n=demo_augmentation_every_n,
                        obs_augmentation_every_n=obs_augmentation_every_n,
                        use_low_dim_pcd=use_low_dim_pcd,
                        use_pcd=use_pcd,
                        use_rgb=use_rgb,
                        use_depth=use_depth,
                        use_mask=use_mask,
                        use_pose=use_pose,
                        keypoints_only=keypoints_only,
                        use_low_dim_state=use_low_dim_state,
                        val_ratio=val_ratio) 

class RLBenchImageDataset(RLBenchDataset, BaseImageDataset):
    def __init__(self,
                    root,
                    cameras=['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
                    image_size=(128, 128),
                    use_pcd=True,
                    use_rgb=True,
                    use_depth=True,
                    use_mask=True,
                    task_name="open_drawer",
                    num_episodes=1,
                    variation=0,
                    n_obs=2,
                    demo_augmentation_every_n=10,
                    obs_augmentation_every_n=10,
                    keypoints_only=False,
                    use_low_dim_state=False,
                    val_ratio=0.0,
                    use_low_dim_pcd=False):
        super().__init__(root=root,
                        task_name=task_name,
                        num_episodes=num_episodes,
                        variation=variation,
                        cameras=cameras,
                        image_size=image_size,
                        n_obs=n_obs,
                        demo_augmentation_every_n=demo_augmentation_every_n,
                        obs_augmentation_every_n=obs_augmentation_every_n,
                        use_low_dim_pcd=use_low_dim_pcd,
                        use_pcd=use_pcd,
                        use_rgb=use_rgb,
                        use_depth=use_depth,
                        use_mask=use_mask,
                        use_pose=False,
                        keypoints_only=keypoints_only,
                        use_low_dim_state=use_low_dim_state,
                        val_ratio=val_ratio)

def test():
    from pytorch3d.transforms import quaternion_to_matrix
    def format_batch(batch):
        kp = batch['obs']['keypoint_poses']
        obs = {f"kp{i}_pos": kp[:,i:i+1,:3,3] for i in range(kp.shape[1])}
        obs.update({f"kp{i}_rot": kp[:,i:i+1,:3,:3] for i in range(kp.shape[1])})
        obs.update({f"robot0_eef_pos": batch['obs']['agent_pose'][:,:,:3,3]})
        obs.update({f"robot0_eef_rot": batch['obs']['agent_pose'][:,:,:3,:3]})
        obs.update({'low_dim_state': batch['obs']['low_dim_state'].unsqueeze(1)})
        act_R = quaternion_to_matrix(torch.cat([batch['action'][:,:,6:7], batch['action'][:,:,3:6]], dim=-1))
        action = {f"act_p": batch['action'][:,:,:3]}
        action.update({f"act_r": act_R})
        action.update({f"act_gr": batch['action'][:,:,7]})
        return {'obs': obs, 'action': action}

    dataset = RLBenchDataset(
        root = "/home/felix/Workspace/diffusion_policy_felix/data/keypoint/train",
        task_name="put_item_in_drawer",
        num_episodes=1,
        variation=0,
        cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
        image_size=(128, 128),
        n_obs=3,
        keypoints_only=False,
        demo_augmentation_every_n=5,
        obs_augmentation_every_n=5,
        use_low_dim_pcd=True,
        use_pcd = False,
        use_rgb = False,
        use_depth = False,
        use_mask = False,
        use_pose = True,
        use_low_dim_state=True,
        val_ratio=0.0
    )

    def print_dict(x, indent=0):
        for k in x.keys():
            if isinstance(x[k], dict):
                print(" "*3*indent+k+":")
                print_dict(x[k], indent+1)
            else:
                print(" "*3*indent+k+":", x[k].detach().cpu().numpy().shape)
                print( x[k][0].detach().cpu().numpy())

    data = dataset[0]
    batch = dict_apply(data, lambda x: x.unsqueeze(0))
    print_dict(batch)
    print("Dataset length:", len(dataset))
    return
    batch = format_batch(batch)
    print_dict(batch)


    from diffusion_policy.common.visualization_se3 import visualize_frames, visualize_poses_and_actions
    from diffusion_policy.model.common.so3_util import quaternion_to_matrix, log_map, exp_map, se3_inverse, apply_transform

    data = dataset[0]
    batch = dict_apply(data, lambda x: x.unsqueeze(0))
    batch = format_batch(batch)

    state_rotation = [v for k, v in batch['obs'].items() if 'rot' in k]
    state_translation = [v for k, v in batch['obs'].items() if 'pos' in k]

    state_rotation = torch.cat(state_rotation, dim=1).flatten(0, 1)
    state_translation = torch.cat(state_translation, dim=1).flatten(0, 1)

    action_rotation = batch['action']['act_r'].flatten(0, 1)
    action_translation = batch['action']['act_p'].flatten(0, 1)
    visualize_poses_and_actions(state_rotation, state_translation, action_rotation, action_translation)

if __name__ == "__main__":
    test()