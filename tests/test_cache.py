from diffusion_policy.dataset.rlbench_dataset import RLBenchNextBestPoseDataset, RLBenchDataset
import os
from diffusion_policy.common.pytorch_util import print_dict, compare_dicts
from torchvision.utils import save_image, make_grid
import torch
import tqdm

task = 'sweep_to_dustpan_of_size'

save_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', 'test_dataset', 'image_keypoints.zarr')
os.makedirs(save_path, exist_ok=True)

dataset = RLBenchNextBestPoseDataset(
    dataset_path=os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image_keypoints.zarr',
    task_name=task,
    n_episodes=-1,
    cache_size=100,
)
val_dataset = dataset.get_validation_dataset()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

dataset.empty_cache()

ds1 = []
for i, data in tqdm.tqdm(enumerate(val_dataloader)):
    print(len(dataset._cache))
    print(len(val_dataset._cache))
    continue

for i, data in tqdm.tqdm(enumerate(val_dataloader)):
    ds1.append(data)


val_dataset.empty_cache()
del data
ds2 = []
for i, data in tqdm.tqdm(enumerate(val_dataloader)):
    ds2.append(data)

print("Checking if the two datasets are the same")
for i in tqdm.tqdm(range(len(ds1))):
    d1 = ds1[i]
    d2 = ds2[i]

    compare_dicts(d1, d2)