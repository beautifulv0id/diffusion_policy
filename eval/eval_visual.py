if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
import torch.utils
import torch.utils.data
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.dataset.rlbench_zarr_dataset import RLBenchNextBestPoseDataset
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config, create_robomimic_from_rlbench_action,mask_out_features_pcd
from diffusion_policy.env_runner.rlbench_utils import _evaluate_task_on_demos
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from diffusion_policy.common.logger_utils import write_video
from diffusion_policy.common.pytorch_util import dict_apply
import tap
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from pathlib import Path
import json
from typing import List
import matplotlib.pyplot as plt
from torch import einsum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion
from torch.nn.functional import interpolate
from PIL import Image
import io
import numpy as np
from torchvision.utils import save_image, make_grid

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'eval')
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.07.10/17.22.29_train_diffuser_actor_sweep_to_dustpan_of_size_mask')
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/image')
    config : str = 'train_diffuser_actor.yaml'
    overrides: List[str] = []


def create_obs_state_plot(obs, gt_action=None, pred_action=None, downsample=1, use_mask=False, lowdim = False, quaternion_format: str = 'wxyz'):
    if lowdim:
        pcd = obs['low_dim_pcd']
        rgb = torch.zeros_like(pcd)
        rgb[..., 0] = 1.0
    else:
        pcd = obs['pcd']
        rgb = obs['rgb']

    pcd = pcd.cpu()
    rgb = rgb.cpu()
    mask = obs['mask'] if use_mask else None
    curr_gripper = obs['curr_gripper'].cpu()
    out = create_robomimic_from_rlbench_action(curr_gripper, quaternion_format=quaternion_format)
    curr_gripper_rot = out['act_r'].squeeze(0).float()
    curr_gripper_pos = out['act_p'].squeeze(0).float()
    curr_gripper_gr = out['act_gr'].squeeze(0).float()

    if gt_action is not None:
        gt_action = gt_action.cpu()
        out = create_robomimic_from_rlbench_action(gt_action, quaternion_format=quaternion_format)
        action_rot1 = out['act_r'].squeeze(0).float()
        action_pos1 = out['act_p'].squeeze(0).float()
        action_gr1 = out['act_gr'].squeeze(0).float()

    if pred_action is not None:
        pred_action = pred_action.cpu()
        out = create_robomimic_from_rlbench_action(pred_action, quaternion_format=quaternion_format)
        action_rot2 = out['act_r'].squeeze(0).float()
        action_pos2 = out['act_p'].squeeze(0).float()
        action_gr2 = out['act_gr'].squeeze(0).float()

    n_obs = curr_gripper_rot.shape[0]
    n_actions = action_rot1.shape[0] if gt_action is not None else 0

    obs_colors = cm.get_cmap('Blues',  n_obs+ 1)
    action_colors = cm.get_cmap('Reds',  n_actions)
    gt_action_colors = cm.get_cmap('Greens',  n_actions)

    def create_gripper_pts(gripper_opens, scale = 0.2):
        gripper_pts = []
        for gripper_open in gripper_opens:
            if not gripper_open:
                gripper_pts.append(torch.tensor([
                    [0, 0, -1],
                    [0, 0, 0],
                    [0, -0.1, 0],
                    [0, 0.1, 0],
                    [0, -0.1, 0],
                    [0, -0.1, 0.5],
                    [0, 0.1, 0],
                    [0, 0.1, 0.5],
                    [0, 0.01, 0.99],
                    [0, -0.01, 1.01],
                    [0, 0.01, 1.01],
                    [0, -0.01, 0.99],
                ]))
            else:
                gripper_pts.append(torch.tensor([
                    [0, 0, -1],
                    [0, 0, 0],
                    [0, -0.5, 0],
                    [0, 0.5, 0],
                    [0, -0.5, 0],
                    [0, -0.5, 0.5],
                    [0, 0.5, 0],
                    [0, 0.5, 0.5],
                    [0, 0.01, 0.99],
                    [0, -0.01, 1.01],
                    [0, 0.01, 1.01],
                    [0, -0.01, 0.99],
                ]) )

        gripper_pts = torch.stack(gripper_pts)
        gripper_pts = gripper_pts + torch.tensor([0, 0, -1]).unsqueeze(0)
        gripper_pts = gripper_pts * scale

        return gripper_pts
    
    def plot_gripper(ax, gripper_pos, gripper_rot, gripper_open, scale = 0.2, colors=obs_colors):
        gripper_pts = create_gripper_pts(gripper_open, scale)
        gripper_pts = einsum('nij,nkj->nki', gripper_rot, gripper_pts)
        gripper_pts = gripper_pts + gripper_pos.unsqueeze(1)
        gripper_pts = gripper_pts.reshape(gripper_pos.shape[0], -1, 3).numpy()
        for i, gripper_pts_i in enumerate(gripper_pts):
            for j in range(0, len(gripper_pts_i), 2):
                line = ax.plot(gripper_pts_i[j:j+2,0], gripper_pts_i[j:j+2,1], gripper_pts_i[j:j+2,2], linewidth=1, zorder=10, color=colors(i+1))
                if j == 0:
                    line[0].set_label(f"Gripper t={i-n_obs+1}")
    
    def plot_action(ax, action_pos, action_rot, gripper_open, scale = 0.1, linewidth=4, colors=action_colors, label_prefix="GT Action t="):
        gripper_pts = create_gripper_pts(gripper_open, scale)
        gripper_pts = einsum('nij,nkj->nki', action_rot, gripper_pts)
        gripper_pts = gripper_pts + action_pos.unsqueeze(1)
        gripper_pts = gripper_pts.reshape(action_pos.shape[0], -1, 3).numpy()
        for i, gripper_pts_i in enumerate(gripper_pts):
            for j in range(0, len(gripper_pts_i), 2):
                line = ax.plot(gripper_pts_i[j:j+2,0], gripper_pts_i[j:j+2,1], gripper_pts_i[j:j+2,2], linewidth=linewidth, zorder=10, color=colors(i+1))
                if j == 0:
                    line[0].set_label(label_prefix + f" t={i+1}")

    def plot_pcd(ax, pcd, rgb, mask=None):
        if not lowdim:
            if downsample > 1:
                b, v, _, h, w = rgb.shape
                h, w = rgb.shape[-2] // downsample, rgb.shape[-1] // downsample

                pcd = interpolate(pcd.reshape((-1,) + pcd.shape[-3:]), size=(h, w), mode='bilinear').reshape(b, v, 3, h, w)
                rgb = interpolate(rgb.reshape((-1,) + rgb.shape[-3:]), size=(h, w), mode='bilinear').reshape(b, v, 3, h, w)
                if mask is not None:
                    mask = interpolate(mask.reshape((-1,) + mask.shape[-3:]).float(), size=(h, w), mode='nearest').bool().reshape(b, v, 1, h, w)

            if mask is not None:
                rgb, pcd = mask_out_features_pcd(mask, rgb, pcd, n_min=1, n_max=1000000)
                rgb = rgb
                pcd = pcd
            else:
                pcd = pcd.permute(0, 1, 3, 4, 2)
                rgb = rgb.permute(0, 1, 3, 4, 2)
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3)
        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c=rgb, s=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45, roll=0)
    RADIUS = .7  # Control this value.
    ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_zlim3d(-RADIUS / 2 + 1, RADIUS / 2 + 1)
    ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
    plot_pcd(ax, pcd, rgb, mask)
    plot_gripper(ax, curr_gripper_pos, curr_gripper_rot, curr_gripper_gr, scale=0.1)
    if gt_action is not None:
        plot_action(ax, action_pos1, action_rot1, action_gr1, label_prefix="GT Action=", colors=gt_action_colors)
    if pred_action is not None:
        plot_action(ax, action_pos2, action_rot2, action_gr2, label_prefix="Pred Action=")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf)).transpose(2, 0, 1)
    buf.close()
    plt.close()

    return image


def load_model(hydra_path, cfg):
    checkpoint_dir = Path(hydra_path).joinpath('checkpoints')
    checkpoint_map = os.path.join(checkpoint_dir, 'checkpoint_map.json')
    if os.path.exists(checkpoint_map):
        with open(checkpoint_map, 'r') as f:
            checkpoint_map = json.load(f)
    checkpoint = Path(sorted(checkpoint_map.items(), key=lambda x: x[1])[0][0]).name
    checkpoint_path = checkpoint_dir.joinpath(checkpoint)
    policy = hydra.utils.instantiate(cfg.policy)
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    return policy

def load_dataset(cfg):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset

def load_config(config_name: str, overrides: list = []):
    with initialize(config_path=Path('../diffusion_policy/config'), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg

def load_overrides(hydra_path, overrides):
    this_overrides = OmegaConf.load(os.path.join(hydra_path, ".hydra", "overrides.yaml"))
    overrides_map = {}
    for override in this_overrides + overrides:
        k, v = override.split('=')
        overrides_map[k] = v
    this_overrides = [k + '=' + v for k, v in overrides_map.items()]
    return this_overrides

if __name__ == '__main__':
    args = Arguments().parse_args()

    overrides = load_overrides(args.hydra_path, args.overrides)

    print("Overrides: ", overrides)
    
    cfg = load_config(args.config, overrides)

    task_str = cfg.task.dataset.task_name
    data_root = args.data_root
    hydra_path = args.hydra_path
    save_path = os.path.join(args.save_root, task_str, hydra_path.split('/')[-1])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(save_path, exist_ok=True)

    dataset : RLBenchNextBestPoseDataset = load_dataset(cfg)

    policy = load_model(hydra_path, cfg)
    policy = policy.to(device)
    policy.eval()
    actioner = Actioner(policy)

    print(f"Hydra path: {hydra_path}")
    print(f"Task: {task_str}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    it = iter(dataloader)
    with torch.no_grad():
        for i in range(10):
            batch = next(it)

            batch = dict_apply(batch, lambda x: x.to(device))

            pred = policy.predict_action(batch['obs'])['rlbench_action'].cpu().detach()

            batch = dict_apply(batch, lambda x: x.cpu().detach())

            print("Pred: ", pred[...,7])
            print("GT: ", batch['action']['gt_trajectory'][...,7])

            img = create_obs_state_plot(batch['obs'], gt_action=batch['action']['gt_trajectory'], pred_action=pred, downsample=1, use_mask=False, lowdim = False, quaternion_format='xyzw')
            save_image(torch.tensor(img).float() / 255, os.path.join(save_path, f'pred_{i}.png'))