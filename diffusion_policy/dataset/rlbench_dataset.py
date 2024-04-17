from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from pathlib import Path
from rlbench.utils import get_stored_demos
from diffusion_policy.common.launch_utils import create_dataset
from rlbench.observation_config import ObservationConfig, CameraConfig

class RLBenchDataset(BaseImageDataset):
    def __init__(self,
            root, 
            task_name="open_drawer",
            num_episodes=1,
            variation=0,
            cameras = ['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'],
            n_obs=2,
            build_history_from_augment_obs=True, # If True, observations are taken from the autmented demo
            demo_augmentation_every_n=10,
            use_task_keypoints=True,
            use_pcd = False,
            use_rgb = False,
            use_depth = False,
            use_mask = False,
            keypoints_only=False,
            val_ratio=0.0,
            ):
        super().__init__()
        # assert variation == 0

        root = Path(root)
        # TODO: change this
        camera_config = CameraConfig(rgb=use_rgb, depth=use_depth, mask=use_mask, point_cloud=use_pcd)
        camera_off_config = CameraConfig(rgb=False, depth=False, mask=False, point_cloud=False)
        all_cameras = [k.split("_camera")[0] for k in ObservationConfig().__dict__.keys() if k.endswith('camera')]
        unused_cameras = [k for k in all_cameras if k not in cameras]
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
        n_train = num_episodes - n_val
        
        val_idxs = np.random.choice(num_episodes, size=n_val, replace=False)
        train_idxs = np.array([i for i in range(num_episodes) if i not in val_idxs])
        train_demos = [demos[i] for i in train_idxs]
        val_demos = [demos[i] for i in val_idxs]    
        
        train_dataset, train_demo_begin = create_dataset(train_demos, 
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        n_obs=n_obs if not build_history_from_augment_obs else 1, 
                                        use_task_keypoints=use_task_keypoints,
                                        use_pcd=use_pcd,
                                        keypoints_only=keypoints_only)
        
        val_dataset, val_demo_begin = create_dataset(val_demos,
                                        cameras=cameras, 
                                        demo_augmentation_every_n=demo_augmentation_every_n, 
                                        n_obs=n_obs if not build_history_from_augment_obs else 1, 
                                        use_task_keypoints=use_task_keypoints,
                                        use_pcd=use_pcd,
                                        keypoints_only=keypoints_only)
        
        self.demos = train_demos
        self.dataset = train_dataset
        self.demo_begin = train_demo_begin
        self.val_demos = val_demos
        self.val_dataset = val_dataset
        self.val_demo_begin = val_demo_begin
        self.n_obs = n_obs
        self.augment_obs = build_history_from_augment_obs  

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.augment_obs:
            data = self.dataset[idx]
        else:
            demo_begin = self.demo_begin[idx]
            data = [self.dataset[max(demo_begin, idx-i)] for i in range(self.n_obs)]

            def merge_dicts(dict_list):
                merged_dict = {}
                for k in dict_list[0].keys():
                    if isinstance(dict_list[0][k], np.ndarray):
                        merged_dict[k] = np.stack([d[k][0] for d in dict_list])
                    elif isinstance(dict_list[0][k], (np.float32, int, np.bool_)):
                        merged_dict[k] = np.array([d[k] for d in dict_list])
                    else:
                        merged_dict[k] = merge_dicts([d[k] for d in dict_list])
                return merged_dict
            data = merge_dicts(data)

        data["action"] = data["action"][-1:]
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.dataset = self.val_dataset
        val_set.demo_begin = self.val_demo_begin
        val_set.demos = self.val_demos
        return val_set

    def get_normalizer(self):
        pass

import hydra
from omegaconf import OmegaConf
import pathlib
OmegaConf.register_new_resolver("eval", eval, replace=True)
import pickle 

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config')),
    config_name='train_diffusion_unet_lowdim_relative_workspace.yaml'
)
def test(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    dataset : RLBenchDataset = hydra.utils.instantiate(cfg.task.dataset)

    def print_dict(x, indent=0):
        for k in x.keys():
            if isinstance(x[k], dict):
                print(" "*3*indent+k+":")
                print_dict(x[k], indent+1)
            else:
                print(" "*3*indent+k+":", x[k].numpy().shape)
    data = dataset[0]
    
    print_dict(data)
    print("Dataset length:", len(dataset))

if __name__ == "__main__":
    test()