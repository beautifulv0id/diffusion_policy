import os
from rlbench.utils import get_stored_demos
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from rlbench.backend.observation import Observation
from rlbench.task_environment import TaskEnvironment
import numpy as np

task_str = 'put_item_in_drawer'
dataset_name = 'image'
variation = 0
data_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', dataset_name)

env = RLBenchEnv(data_path=data_path, headless=True)

obs_config = create_obs_config(image_size=[128, 128], apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=True)
demo = get_stored_demos(amount=1, dataset_root=data_path, task_name=task_str, variation_number=variation, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)[0]

obs : Observation = demo[0]

print(obs.misc.keys())
