from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply, print_dict
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseLowdimDataset
from pathlib import Path
from rlbench.utils import get_stored_demos
from diffusion_policy.common.rlbench_util import create_dataset, create_obs_config, create_obs_state_plot, CAMERAS
from rlbench.observation_config import ObservationConfig, CameraConfig
from PIL import Image

MASK_IDS_PERACT = {
    # "open_drawer" : [0, 200] # from 0 to 200
    "open_drawer" : [305, 319], #[65473, 65487],
    "put_item_in_drawer" : [65473, 65487],
}

MASK_IDS = {
    "open_drawer" : [704, 718],
}

# peract: 
# open_drawer: 65474 65477 65479 65484 65486

# mine:
# open_drawer: 105 107 112 114 117


class RLBenchDataset():
    def __init__(self,
            root, 
            task_name="open_drawer",
            num_episodes=1,
            variation=0,
            cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
            image_size=(128, 128),
            n_obs_steps=2,
            horizon=1,
            use_keyframe_actions=True,
            demo_augmentation_every_n=1,
            obs_augmentation_every_n=10,
            use_keyframe_observations=True,
            use_low_dim_pcd=False,
            use_pcd = False,
            use_rgb = False,
            use_depth = False,
            use_mask = False,
            object_ids = None,
            use_pose = False,
            use_low_dim_state=False,
            val_ratio=0.0,
            ):

        root = Path(root)
        obs_config = create_obs_config(image_size=image_size,
                                       apply_rgb=use_rgb,
                                       apply_depth=use_depth,
                                       apply_pc=use_pcd,
                                       apply_mask=use_mask,
                                       apply_cameras=cameras)

        # if use_mask:
        #     mask_ids = MASK_IDS[task_name]
        # else:
        #     mask_ids = None

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
                                        n_obs=n_obs_steps, 
                                        use_low_dim_pcd=use_low_dim_pcd,
                                        use_pcd=use_pcd,
                                        use_rgb=use_rgb,
                                        use_pose=use_pose,
                                        # use_depth=use_depth,
                                        use_mask=use_mask,
                                        mask_ids=object_ids,
                                        horizon=horizon,
                                        use_keyframe_actions=use_keyframe_actions,
                                        use_low_dim_state=use_low_dim_state,
                                        use_keyframe_observations=use_keyframe_observations)
        
        val_dataset, val_demo_begin = create_dataset(val_demos,
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        obs_augmentation_every_n=obs_augmentation_every_n,
                                        n_obs=n_obs_steps, 
                                        use_low_dim_pcd=use_low_dim_pcd,
                                        use_pcd=use_pcd,
                                        use_rgb=use_rgb,
                                        use_pose=use_pose,
                                        # use_depth=use_depth,
                                        use_mask=use_mask,
                                        mask_ids=object_ids,
                                        horizon=horizon,
                                        use_keyframe_actions=use_keyframe_actions,
                                        use_low_dim_state=use_low_dim_state,
                                        use_keyframe_observations=use_keyframe_observations)
        
        self.demos = train_demos
        self.dataset = train_dataset
        self.demo_begin = train_demo_begin
        self.val_demos = val_demos
        self.val_dataset = val_dataset
        self.val_demo_begin = val_demo_begin
        self.n_obs = n_obs_steps

        batch = self[40]
        batch = dict_apply(batch, lambda x: x.unsqueeze(0))
        img = create_obs_state_plot(batch['obs'], use_mask=True)
        img = img.transpose(1,2,0)[:,:,:3]
        img = Image.fromarray(img)
        img.save("/home/felix/Workspace/diffusion_policy_felix/run_obs_img.png")

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
        this_data['obs']['curr_gripper'] = data['obs']['curr_gripper']

        if 'low_dim_state' in data['obs']:
            this_data['obs']['low_dim_state'] = data['obs']['low_dim_state']

        if 'rgb' in data['obs']:
            this_data['obs']['rgb'] = data['obs']['rgb'][-1]
        
        if 'pcd' in data['obs']:   
            this_data['obs']['pcd'] = data['obs']['pcd'][-1]

        if 'mask' in data['obs']:
            this_data['obs']['mask'] = data['obs']['mask'][-1]

        for k in data['obs'].keys():
            if 'robot0_eef_rot' in k or \
            'robot0_eef_pos' in k or \
            'curr_gripper' in k or \
            'low_dim_state' in k or \
            'rgb' in k or \
            'pcd' in k or \
            'mask' in k:
                continue
            this_data['obs'][k] = data['obs'][k][-1:]

        torch_data = dict_apply(this_data, torch.from_numpy)
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
                    use_keyframe_actions=True,
                    use_keyframe_observations=False,
                    n_obs_steps=2,
                    horizon=1,
                    demo_augmentation_every_n=1,
                    obs_augmentation_every_n=10,
                    use_low_dim_state=True,
                    val_ratio=0.0):
        super().__init__(root=root,
                        task_name=task_name,
                        num_episodes=num_episodes,
                        variation=variation,
                        cameras=cameras,
                        image_size=image_size,
                        n_obs_steps=n_obs_steps,
                        horizon=horizon,
                        use_keyframe_actions=use_keyframe_actions,
                        use_keyframe_observations=use_keyframe_observations,
                        demo_augmentation_every_n=demo_augmentation_every_n,
                        obs_augmentation_every_n=obs_augmentation_every_n,
                        use_low_dim_pcd=use_low_dim_pcd,
                        use_pcd=False,
                        use_rgb=False,
                        use_depth=False,
                        use_mask=False,
                        use_pose=use_pose,
                        use_low_dim_state=use_low_dim_state,
                        val_ratio=val_ratio) 

class RLBenchImageDataset(RLBenchDataset, BaseImageDataset):
    def __init__(self,
                    root,
                    cameras=['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
                    image_size=(128, 128),
                    use_pcd=True,
                    use_rgb=True,
                    use_depth=False,
                    use_mask=False,
                    task_name="open_drawer",
                    num_episodes=1,
                    variation=0,
                    n_obs_steps=2,
                    horizon=1,
                    use_keyframe_actions=True,
                    use_keyframe_observations=False,
                    demo_augmentation_every_n=1,
                    obs_augmentation_every_n=10,
                    use_low_dim_state=False,
                    val_ratio=0.0,
                    **kwargs):
        super().__init__(root=root,
                        task_name=task_name,
                        num_episodes=num_episodes,
                        variation=variation,
                        cameras=cameras,
                        image_size=image_size,
                        n_obs_steps=n_obs_steps,
                        horizon=horizon,
                        use_keyframe_actions=use_keyframe_actions,
                        use_keyframe_observations=use_keyframe_observations,
                        demo_augmentation_every_n=demo_augmentation_every_n,
                        obs_augmentation_every_n=obs_augmentation_every_n,
                        use_low_dim_pcd=False,
                        use_pcd=use_pcd,
                        use_rgb=use_rgb,
                        use_depth=use_depth,
                        use_mask=use_mask,
                        use_pose=False,
                        use_low_dim_state=use_low_dim_state,
                        val_ratio=val_ratio,
                        **kwargs)
        
    

def test():
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion
    from diffusion_policy.common.rlbench_util import create_rlbench_action
    from diffusion_policy.env.rlbench.rlbench_env import visualize
    from PIL import Image

    dataset = RLBenchImageDataset(
        root = "/home/felix/Workspace/diffusion_policy_felix/data/images/",
        task_name="open_drawer",
        num_episodes=1,
        variation=0,
        cameras = ['left_shoulder', 'right_shoulder', 'front', 'overhead', 'wrist'],
        image_size=(256, 256),
        n_obs_steps=3,
        horizon=1,
        use_mask=True,
        use_keyframe_actions=True,
        use_keyframe_observations=True,
        demo_augmentation_every_n=1,
        obs_augmentation_every_n=10,
        use_low_dim_state=False,
        object_ids = MASK_IDS['open_drawer'],
        val_ratio=0
    )

    data = dataset[60]
    batch = dict_apply(data, lambda x: x.unsqueeze(0))
    print("Dataset length:", len(dataset))
    print_dict(batch)
    print(len(dataset))

    img = create_obs_state_plot(batch['obs'], use_mask=True)
    img = img.transpose(1,2,0)[:,:,:3]
    img = Image.fromarray(img)
    img.save("obs_img.png")
    return

    # visualize(batch['obs'], batch['action'])
    import matplotlib.pyplot as plt
    def save_img(img, mask):
        img = torch.where(mask, img, torch.zeros_like(img))
        plt.imshow(img.permute(1,2,0))
        plt.savefig(f"img_{view}.png")

    view = 2

    # obj_ids = batch['obs']['mask'][0,view].int()
    # mask = torch.zeros_like(obj_ids)
    # # for id in MASK_IDS['open_drawer']:
    # #     mask = mask | (obj_ids == id)
    # mask = (obj_ids > 100) & (obj_ids < 280)
    # mask = mask.bool()
    img = batch['obs']['rgb'][0,view]
    mask = batch['obs']['mask'][0,view]
    save_img(img, mask)



    # def save_mask_for_id(mask, rgb, id):
    #     import matplotlib.pyplot as plt
    #     mask = mask.int() == id
    #     img = torch.where(mask, rgb, torch.zeros_like(rgb))
    #     plt.imshow(img.permute(1,2,0))
    #     plt.savefig(f"mask_{id}.png")

    # for i in torch.unique(batch['obs']['mask'][0].int()):
    #     save_mask_for_id(batch['obs']['mask'][0,view], batch['obs']['rgb'][0,view], i)

    
    return 

    from diffusion_policy.common.visualization_se3 import visualize_frames, visualize_poses_and_actions
    from diffusion_policy.model.common.so3_util import quaternion_to_matrix, log_map, exp_map, se3_inverse, apply_transform

    state_rotation = [v for k, v in batch['obs'].items() if 'rot' in k]
    state_translation = [v for k, v in batch['obs'].items() if 'pos' in k]

    state_rotation = torch.cat(state_rotation, dim=1).flatten(0, 1)
    state_translation = torch.cat(state_translation, dim=1).flatten(0, 1)

    action_rotation = batch['action']['act_r'].flatten(0, 1)
    action_translation = batch['action']['act_p'].flatten(0, 1)
    visualize_poses_and_actions(state_rotation, state_translation, action_rotation, action_translation)

if __name__ == "__main__":
    test()
