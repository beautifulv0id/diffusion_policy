from diffusion_policy.dataset.rlbench_next_best_pose_dataset import RLBenchNextBestPoseDataset
import os
from diffusion_policy.common.pytorch_util import print_dict

dataset = RLBenchNextBestPoseDataset(
    dataset_path=os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image.zarr',
    task_name='open_drawer',
    n_episodes=1,
)

print(len(dataset))

print_dict(dataset[0])