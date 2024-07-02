if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.env_runner.rlbench_utils import _evaluate_task_on_demos
from diffusion_policy.env.rlbench.rlbench_utils import Mover, Actioner, task_file_to_task_class, get_actions_from_demo
from diffusion_policy.common.logger_utils import write_video

class ReplayPolicy:
    def __init__(self, demo):
        self._actions = get_actions_from_demo(demo)
        self.idx = 0
        self.n_obs_steps = 1

    def predict_action(self, obs):
        action = self._actions[self.idx % len(self._actions)]
        self.idx += 1
        return {
            'rlbench_action' : 
                action.cpu().numpy()
        }
    
    def eval(self):
        pass

    def parameters(self):
        return iter([torch.empty(0)])


data_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'peract', 'raw')
save_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', os.path.basename(__file__).replace('.py', ''))
task_str = 'open_drawer'

if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)

    runner = RLBenchRunner(
        output_dir=save_path,
        data_root=data_path,
        headless=True,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        task_str=task_str,
        max_steps=3,
        max_episodes=1,
    )

    
    obs_config = create_obs_config(image_size=[128, 128], apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
    demos = get_stored_demos(amount=1, dataset_root=data_path, task_name=task_str, variation_number=0, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)

    actioner = Actioner(
        policy=ReplayPolicy(demos[0]),
        action_dim=8,
    )

    log_data = _evaluate_task_on_demos(env_args=runner.env_args,
                            task_str=runner.task_str,
                            demos=demos[:1],
                            max_steps=3,
                            actioner=actioner,
                            max_rrt_tries=1,
                            demo_tries=1,
                            n_visualize=1,
                            verbose=True)

    rgbs_ls = log_data.pop("rgbs")
    obs_state_ls = log_data.pop("obs_state")
    mask_ls = log_data.pop("mask")

    if len(rgbs_ls) > 0:
        rgbs = rgbs_ls[0].transpose(0, 2, 3, 1)
        write_video(rgbs, os.path.join(save_path, "rgbs.mp4"), fps=30)
    else:
        print("No rgbs")

    if len(obs_state_ls) > 0:
        obs_state = obs_state_ls[0].transpose(0, 2, 3, 1)
        write_video(obs_state, os.path.join(save_path, "obs_state.mp4"), fps=1)
    else:
        print("No obs")

    if len(mask_ls) > 0:
        mask = mask_ls[0].transpose(0, 2, 3, 1)
        write_video(mask, os.path.join(save_path, "mask.mp4"), fps=1)
    else:
        print("No mask")
    