from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from rlbench.task_environment import TaskEnvironment
from rlbench.demo import Demo
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class
from typing import List
import wandb

class RLBenchRunner(BaseLowdimRunner):
    def __init__(self, 
                 output_dir,
                 data_root, #TODO: or pass demos?
                task_str: str,
                max_steps: int,
                max_episodes: int,
                max_rtt_tries: int = 1,
                demo_tries: int = 1,
                headless: bool = True,
                collision_checking: bool = True,
                action_dim: int = 8,
                n_train_vis: int = 1,
                n_val_vis: int = 1,
                obs_history_augmentation_every_n: int = 1,
                image_size=(128, 128),
                render_image_size=(128, 128),
                apply_rgb=False,
                apply_depth=False,
                apply_pc=False,
                apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
                 ):
        super(RLBenchRunner, self).__init__(output_dir)
        self.task_str = task_str

        env = RLBenchEnv(data_path=data_root, 
                            image_size=image_size,
                            render_image_size=render_image_size,
                            headless=headless, 
                            apply_rgb=apply_rgb,
                            apply_depth=apply_depth,
                            apply_pc=apply_pc,
                            apply_cameras=apply_cameras,
                            collision_checking=collision_checking,
                            obs_history_augmentation_every_n=obs_history_augmentation_every_n)        
        self.task_str = task_str
        self.env = env
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.max_rtt_tries = max_rtt_tries
        self.demo_tries = demo_tries
        self.n_train_vis = n_train_vis
        self.n_val_vis = n_val_vis
        self.max_episodes = max_episodes

    def run(self, policy: BaseLowdimPolicy, demos: List[Demo], mode: str = "train") -> Dict:
        actioner = Actioner(policy=policy, action_dim=self.action_dim)

        if mode == "train":
            n_vis = self.n_train_vis
        elif mode == "eval":
            n_vis = self.n_val_vis

        if len(demos) == 0:
            return {}
        
        task_type = task_file_to_task_class(self.task_str)
        task = self.env.env.get_task(task_type)
        self.env.launch()
        log_data = self.env._evaluate_task_on_demos(
            demos=demos[:self.max_episodes],
            task_str=self.task_str,
            task=task,
            max_steps=self.max_steps,
            actioner=actioner,
            max_rtt_tries=self.max_rtt_tries,
            demo_tries=self.demo_tries,
            verbose=False,
            n_visualize=n_vis,
            num_history=policy.n_obs_steps,
        )

        name = mode + "_success_rate"
        log_data[name] = log_data.pop("success_rate")
        rgbs_ls = log_data.pop("rgbs_ls")

        for i, rgbs in enumerate(rgbs_ls):
            if rgbs is not None:
                sim_video = wandb.Video(rgbs, fps=30, format="mp4")
                name = f"video/{mode}_{self.task_str}_{i}"
                log_data[name] = sim_video
        
        return log_data
    
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy
import pathlib

OmegaConf.register_new_resolver("eval", eval, replace=True)

def test():
    import torch
    from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_flow_matching_unet_lowdim_workspace.yaml", overrides=["task=open_drawer_lowdim"])

    OmegaConf.resolve(cfg)
    policy : DiffusionUnetLowDimRelativePolicy = hydra.utils.instantiate(cfg.policy)
    checkpoint_path = "/home/felix/Workspace/diffusion_policy_felix/data/outputs/2024.04.16/17.50.13_flow_matching_unet_lowdim_policy_open_drawer/checkpoints/epoch=465800-val_loss=0.000.ckpt"
    checkpoint = torch.load(checkpoint_path)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()

    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    env_runner = RLBenchRunner(output_dir=str(pathlib.Path(__file__).parent.parent.parent / "data"),
                                data_root=cfg.task.dataset.root,
                                task_str=cfg.task.dataset.task_name,
                                max_steps=3,
                                demos=dataset.demos, 
                                eval_demos=val_dataset.demos,
                                render_image_size=[1280, 720],
                                max_episodes=1,
                                headless=False)

    env_runner.n_val_vis = 0
    env_runner.n_train_vis = 0

    results = env_runner.run(policy)
    print(results)


def test_replay():
    import torch
    import numpy as np
    from diffusion_policy.env.rlbench.rlbench_utils import Actioner, task_file_to_task_class, keypoint_discovery, get_actions_from_demo
    from rlbench.utils import get_stored_demos
    import os

    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/peract"
    task_str = "open_drawer"
    env_runner = RLBenchRunner(output_dir="/home/felix/Workspace/diffusion_policy_felix/data/videos/rlbench_runner_test",
                                data_root=data_path,
                                task_str=task_str,
                                max_steps=3,
                                max_episodes=1,
                                headless=False,
                                apply_rgb=True,
                                apply_depth=False,
                                apply_pc=True)
    
    env = env_runner.env
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    variation = 0
    task.set_variation(variation)
    demos = env.get_demo(task_str, variation, episode_index=0)

    class ReplayPolicy:
        def __init__(self, demo):
            self._actions = get_actions_from_demo(demo)
            print(len(self._actions))
            self.idx = 0
            self.n_obs_steps = 2

        def predict_action(self, obs):
            return {
                "action": self._actions.pop(0).unsqueeze(0),
            }
        
        def eval(self):
            pass

        def parameters(self):
            return iter([torch.empty(0)])

    policy = ReplayPolicy(demos[0])
    env_runner.n_val_vis = 0
    env_runner.n_train_vis = 0
    results = env_runner.run(policy, demos)
    print(results)

if __name__ == "__main__":
    test_replay()
    print("Done!")