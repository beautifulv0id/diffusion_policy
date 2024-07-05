if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from rlbench.demo import Demo
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from typing import List
import wandb
from diffusion_policy.env_runner.rlbench_utils import _evaluate_task_on_demos


class RLBenchRunner(BaseImageRunner):
    def __init__(self, 
                 output_dir,
                 data_root, #TODO: or pass demos?
                task_str: str,
                max_steps: int,
                max_episodes: int,
                max_rrt_tries: int = 1,
                n_action_steps: int = 1,
                demo_tries: int = 1,
                headless: bool = True,
                collision_checking: bool = True,
                action_dim: int = 8,
                n_train_vis: int = 1,
                n_val_vis: int = 1,
                obs_history_from_planner: bool = False,
                obs_history_augmentation_every_n: int = 1,
                n_obs_steps: int = 2,
                image_size=(128, 128),
                render_image_size=(128, 128),
                n_procs_max=1,
                apply_rgb=False,
                apply_depth=False,
                apply_pc=False,
                apply_mask=False,
                apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
                apply_low_dim_pcd=False,
                apply_pose=False,
                adaptor=None,
                 ):
        super(RLBenchRunner, self).__init__(output_dir)
        self.task_str = task_str

        assert any([apply_rgb, apply_pc, apply_low_dim_pcd]), "At least one of apply_rgb, apply_pc, apply_low_dim_pcd must be True"
        
        self.env_args = {
            "data_path" : data_root, 
            "image_size" : image_size,
            "render_image_size" : render_image_size,
            "headless" : headless, 
            "apply_rgb" : apply_rgb,
            "apply_depth" : apply_depth,
            "apply_pc" : apply_pc,
            "apply_cameras" : apply_cameras,
            "apply_low_dim_pcd" : apply_low_dim_pcd,
            "apply_pose" : apply_pose,
            "apply_mask" : apply_mask,
            "collision_checking" : collision_checking,
            "obs_history_from_planner" : obs_history_from_planner,
            "obs_history_augmentation_every_n" : obs_history_augmentation_every_n,
            "adaptor" : adaptor,
            "n_obs_steps" : n_obs_steps,
            "n_action_steps" : n_action_steps,
        }


        self.task_str = task_str
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.max_rrt_tries = max_rrt_tries
        self.demo_tries = demo_tries
        self.n_train_vis = n_train_vis
        self.n_val_vis = n_val_vis
        self.max_episodes = max_episodes
        self.n_procs_max = n_procs_max

    def run(self, policy: BaseLowdimPolicy, demos: List[Demo], mode: str = "train") -> Dict:
        actioner = Actioner(policy=policy, action_dim=self.action_dim)

        if mode == "train":
            n_vis = self.n_train_vis
        elif mode == "eval":
            n_vis = self.n_val_vis

        if len(demos) == 0:
            return {}
        
        log_data = _evaluate_task_on_demos(env_args=self.env_args,
                                task_str=self.task_str,
                                demos=demos[:self.max_episodes],
                                max_steps=self.max_steps,
                                actioner=actioner,
                                max_rrt_tries=self.max_rrt_tries,
                                demo_tries=self.demo_tries,
                                n_visualize=n_vis,
                                verbose=True,
                                n_procs_max=self.n_procs_max,
                                )


        name = mode + "_success_rate"
        log_data[name] = log_data.pop("success_rate")
        rgbs_ls = log_data.pop("rgbs")
        obs_state_ls = log_data.pop("obs_state")
        mask_ls = log_data.pop("mask")

        for i, rgbs in enumerate(rgbs_ls):
            if rgbs is not None:
                sim_video = wandb.Video(rgbs, fps=30, format="mp4")
                name = f"video/{mode}_{self.task_str}_{i}"
                log_data[name] = sim_video
        
        for i, obs_state in enumerate(obs_state_ls):
            if obs_state is not None:
                obs_state = wandb.Video(obs_state, fps=1, format="mp4")
                name = f"obs_state/{mode}_{self.task_str}_obs_state_{i}"
                log_data[name] = obs_state
        
        for i, mask in enumerate(mask_ls):
            if mask is not None:
                sim_plots = wandb.Video(mask, fps=1, format="mp4")
                name = f"mask/{mode}_{self.task_str}_mask_{i}"
                log_data[name] = sim_plots

        
        return log_data