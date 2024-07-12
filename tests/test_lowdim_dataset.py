from diffusion_policy.dataset.rlbench_zarr_dataset import RLBenchLowDimNextBestPoseDataset
import os
from diffusion_policy.common.pytorch_util import print_dict
from torchvision.utils import save_image, make_grid

tasks = ['open_drawer_keypoint', 'put_item_in_drawer', 'stack_blocks']

dataset_name = 'lowdim_keypoints.zarr'
save_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', 'test_lowdim_dataset', dataset_name)
os.makedirs(save_path, exist_ok=True)
for task in tasks:
    dataset = RLBenchLowDimNextBestPoseDataset(
        dataset_path=os.environ['DIFFUSION_POLICY_ROOT'] + f'/data/{dataset_name}',
        task_name=task,
        n_episodes=-1,
    )

    img = dataset.get_data_visualization(8)
    grid = make_grid(img, nrow=4)
    save_image(grid, os.path.join(save_path, task + '.png'))

