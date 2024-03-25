from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from rlbench.task_environment import TaskEnvironment
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from diffusion_policy.env.rlbench.rlbench_lowdim_env import RLBenchLowDimEnv
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class


class RLBenchLowdimRunner(BaseLowdimRunner):
    def __init__(self, 
                 output_dir,
                 data_root, #TODO: or pass demos?
                task_str: str,
                max_steps: int,
                variation: int,
                num_demos: int,
                max_tries: int = 1,
                headless: bool = False,
                collision_checking: bool = True,
                action_dim: int = 8,
                obs_history_augmentation_every_n: int = 1
                 ):
        super(RLBenchLowdimRunner, self).__init__(output_dir)
        self.task_str = task_str

        env = RLBenchLowDimEnv(data_path=data_root, 
                                    headless=headless, 
                                    collision_checking=collision_checking,
                                    obs_history_augmentation_every_n=obs_history_augmentation_every_n)
        task_type = task_file_to_task_class(task_str)
        task = env.env.get_task(task_type)
        task.set_variation(variation)


        self.task = task
        self.env = env
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.num_demos = num_demos
        self.max_tries = max_tries
        self.variation = variation

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        actioner = Actioner(policy=policy, action_dim=self.action_dim)

        success_rate, _, _ = self.env._evaluate_task_on_one_variation(
            task_str=self.task_str,
            task=self.task,
            max_steps=self.max_steps,
            variation=self.variation,
            num_demos=self.num_demos,
            actioner=actioner,
            max_tries=self.max_tries,
            verbose=False,
            num_history=policy.n_obs_steps,
        )
        
        return {"success_rate": success_rate}

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

def test():
    import torch
    from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="train_diffusion_unet_lowdim_relative_workspace")

    OmegaConf.resolve(cfg)
    policy : DiffusionUnetLowDimRelativePolicy = hydra.utils.instantiate(cfg.policy)
    checkpoint_path = "/home/felix/Workspace/diffusion_policy_felix/data/outputs/2024.03.25/09.01.10_diffusion_unet_lowdim_relative_policy_open_drawer/checkpoints/latest.ckpt"
    checkpoint = torch.load(checkpoint_path)

    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    env_runner : RLBenchLowdimRunner
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=".")

    results = env_runner.run(policy)
    print(results)

if __name__ == "__main__":
    test()
    print("Done!")