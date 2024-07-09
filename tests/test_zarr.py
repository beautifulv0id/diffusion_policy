import zarr
import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_name = 'image_keypoints.zarr'
data_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', dataset_name)
dataset = zarr.open(data_path, mode='r')
print(dataset.tree())

save_path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', 'test_zarr', dataset_name)
os.makedirs(save_path, exist_ok=True)

task_root = dataset['open_drawer']
data_root = task_root['data']

for key in data_root.keys():
    data = data_root[key]
    
    if key.endswith('mask'):
        img = data[0]
        img = img.transpose(1, 2, 0) * 255
        img = Image.fromarray(img.squeeze().astype('uint8'))
        img.save(os.path.join(save_path, f'{key}.png'))
    elif key.endswith('rgb'):
        img = data[0]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        img.save(os.path.join(save_path, f'{key}.png'))
