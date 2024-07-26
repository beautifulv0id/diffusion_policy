if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.policy.diffuser_actor import DiffuserActor
from diffusion_policy.dataset.rlbench_dataset import RLBenchNextBestPoseDataset
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.common.pytorch_util import dict_apply, print_dict
import tap
import numpy as np
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from pathlib import Path
import json
from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import einops

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'eval')
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.07.16/19.27.52_train_diffuser_actor_stack_blocks_mask')
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/image')
    config : str = 'train_diffuser_actor.yaml'
    overrides: List[str] = []
    render_image_size: tuple = (256, 256)
    n_demos: int = -1
    demos_from_path: bool = False

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
    with initialize(config_path='../diffusion_policy/config', version_base=None):
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

def extract_pcd_rgb(obs, scale_factor):
    pcd = obs['pcd']
    rgb = obs['rgb']
    b,n,c,h_in, w_in= pcd.shape
    h, w = h_in // scale_factor, w_in // scale_factor
    pcd = F.interpolate(
                pcd.flatten(0, 1),
                (h, w),
                mode='nearest'
            )
    pcd = einops.rearrange(pcd, '(b npts) c h w -> b (npts h w) c', b = b).cpu().numpy()
    rgb = F.interpolate(
                rgb.flatten(0, 1),
                (h, w),
                mode='bilinear'
            )
    rgb = einops.rearrange(rgb, '(b npts) c h w -> b (npts h w) c', b = b).cpu().numpy()
    return pcd[0], rgb[0]

if __name__ == '__main__':
    args = Arguments().parse_args()

    overrides = load_overrides(args.hydra_path, args.overrides)
    print("Overrides: ", overrides)
    
    cfg = load_config(args.config, overrides)
    print("Config loaded")

    save_path = os.path.join(args.save_root, cfg.task.task_name, args.hydra_path.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)

    dataset : RLBenchNextBestPoseDataset = load_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print("Dataset loaded")


    policy : DiffuserActor = load_model(args.hydra_path, cfg)
    policy = policy.to("cuda")
    policy.eval()
    print("Model loaded")

    cvals  = [-1, 0, 0.5,  1]
    colors = [(0.9, 0.9, 0.9, 0.01), (0.5, 0.5, 0.5, 0.01),"salmon","red"]

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    for i in range(4):

        batch = dict_apply(dataset[i], lambda x: x.unsqueeze(0).cuda())

        print("Predicting action")
        with torch.no_grad():
            output = policy.predict_action(batch['obs'], need_attn_weights=True)
        print("Action predicted")

        attn_weights = output['attn_weights'].cpu().numpy() 
        attn_weights = attn_weights / attn_weights.max(axis=-1, keepdims=True)
        attn_pcd = policy.unnormalize_pos(output['attn_pcd']).squeeze().cpu().numpy()

        scale_factor = policy.encoder.downscaling_factor_pyramid[0]
        pcd, rgb = extract_pcd_rgb(batch['obs'], scale_factor)


        c = np.full_like(pcd[:,0], -1.0)
        if policy.use_mask:
            idx = output['mask_idx'][1].cpu().numpy()
            c[idx] = attn_weights.flatten()
        else:
            c = attn_weights

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=0, roll=0)
        cb = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=c, cmap=cmap)
        fig.colorbar(cb)
        fig.savefig(os.path.join(save_path, f'attn_pcd_{i}.png'), dpi=300)
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=0, roll=0)
        cb = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=rgb)
        fig.savefig(os.path.join(save_path, f'rgb_pcd_{i}.png'), dpi=300)
        plt.clf()