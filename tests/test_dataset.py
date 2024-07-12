from diffusion_policy.dataset.rlbench_zarr_dataset import RLBenchNextBestPoseDataset
import os
from diffusion_policy.common.pytorch_util import print_dict
from torchvision.utils import save_image, make_grid

tasks = ['open_drawer', 'put_item_in_drawer', 'stack_blocks', 'sweep_to_dustpan_of_size', 'turn_tap']
tasks = ['open_drawer']

save_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', 'test_dataset', 'image_keypoints.zarr')
os.makedirs(save_path, exist_ok=True)

rot_noise_scale = 0.05
pos_noise_scale = 0.01

for task in tasks:
    dataset = RLBenchNextBestPoseDataset(
        dataset_path=os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image_keypoints.zarr',
        task_name=task,
        n_episodes=-1,
        use_mask=False,
        rot_noise_scale=rot_noise_scale,
        pos_noise_scale=pos_noise_scale,
        image_rescale=(0.5, 1.5)
    )
    img = dataset.get_data_visualization(8)
    grid = make_grid(img, nrow=4)
    save_image(grid, os.path.join(save_path, task + f'_r{rot_noise_scale}_p{pos_noise_scale}.png'))

