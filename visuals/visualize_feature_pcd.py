from diffusion_policy.model.obs_encoders.diffuser_actor_encoder import DiffuserActorEncoder
from diffusion_policy.dataset.rlbench_dataset import RLBenchNextBestPoseDataset
from diffusion_policy.common.pytorch_util import dict_apply, dict_apply_reduce
import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
import numpy as np
import einops
from diffusion_policy.model.vision.clip_wrapper import load_clip
from diffusion_policy.model.vision.resnet import load_resnet50, load_resnet18
from PIL import Image
import io
from torchvision.utils import save_image, make_grid

dataset_path = os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image_keypoints.zarr'
task_name = "put_item_in_drawer"
dataset = RLBenchNextBestPoseDataset(
    dataset_path=os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image_keypoints.zarr',
    task_name=task_name,
    cameras=['left_shoulder', 'right_shoulder', 'wrist', 'overhead', 'front'],
    n_episodes=-1,
    use_mask=False,
    rot_noise_scale=0,
    pos_noise_scale=0,
    image_rescale=(1.0, 1.0)
)

save_path = os.environ['DIFFUSION_POLICY_ROOT'] + f'/data/visuals/feature_pcd/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = DiffuserActorEncoder(embedding_dim=64).to(device)
clip, normalize = load_resnet50()
clip = clip.to(device)
clip.eval()

batch = dict_apply(dataset[8], lambda x: x.unsqueeze(0))

rgb = batch['obs']['rgb']
pcd = batch['obs']['pcd']

rgb = einops.rearrange(rgb, 'b v c h w -> (b v h w) c') 
pcd = einops.rearrange(pcd, 'b v c h w -> (b v h w) c')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=30, roll=0)
RADIUS = 0.75  # Control this value.
ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
ax.set_zlim3d(-RADIUS / 2 + 1, RADIUS / 2 + 1)
ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
plt.gca().set_axis_off()

ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=rgb, s=8)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img.save(os.path.join(save_path, task_name + '_pcd_rgb.png'))
buf.close()
plt.close()

batch = dict_apply(batch, lambda x: x.to(device))
rgb = batch['obs']['rgb']
pcd = batch['obs']['pcd']

rgb = einops.rearrange(rgb, 'b v c h w -> (b v) c h w')
pcd = einops.rearrange(pcd, 'b v c h w -> (b v) c h w')
with torch.no_grad():
    features = clip(normalize(rgb))['res1']


feat_h, feat_w = features.shape[-2:]
pcd = F.interpolate(
    pcd,
    (feat_h, feat_w),
    mode='nearest'
)

features = einops.rearrange(features, 'bv c h w -> (bv h w) c')
pcd = einops.rearrange(pcd, 'bv c h w -> (bv h w) c')

features = features.detach().cpu().numpy()
pcd = pcd.detach().cpu().numpy()

pca = PCA(n_components=3)

features_pca = pca.fit_transform(features)
features_pca = (features_pca - np.min(features_pca, axis=0)) / (np.max(features_pca, axis=0) - np.min(features_pca, axis=0))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=30, roll=0)
RADIUS = 0.75  # Control this value.
ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
ax.set_zlim3d(-RADIUS / 2 + 1, RADIUS / 2 + 1)
ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
ax.grid(False)
plt.gca().set_axis_off()

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=features_pca, s=8)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img.save(os.path.join(save_path, task_name + '_feature_pcd.png'))
buf.close()
plt.close()


