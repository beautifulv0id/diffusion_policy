import os
from rlbench.utils import get_stored_demos
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class
from rlbench.task_environment import TaskEnvironment

task_str = 'open_drawer'
data_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'test_dataset')

env = RLBenchEnv(data_path='', headless=True)

obs_config = create_obs_config(image_size=[128, 128], apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
demo = get_stored_demos(amount=1, dataset_root=data_path, task_name=task_str, variation_number=-1, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)[0]
task_type = task_file_to_task_class(task_str)
task : TaskEnvironment = env.env.get_task(task_type)
task.set_variation(0)

descriptions, observation = task.reset_to_demo(demo)

print('Success!')
    