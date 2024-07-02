import os
from rlbench.utils import get_stored_demos
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class
from rlbench.task_environment import TaskEnvironment
import numpy as np

task_str = 'open_drawer'
variation = -1
data_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'test_dataset')

env = RLBenchEnv(data_path=data_path, headless=True)

sr, demo_valid, success_rates = env.verify_demos(
    task_str=task_str,
    variation=variation,
    num_demos=1,
    max_rrt_tries=10,
    demo_consistency_tries=10,
    verbose=True,
)
print("Success rate: ", sr)
print("Valid demos: ", np.count_nonzero(demo_valid))
print("Invalid demos: ", np.count_nonzero(~demo_valid))
print("Success rates: ", success_rates)
    