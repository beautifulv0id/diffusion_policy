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
from diffusion_policy.common.rlbench_util import create_obs_state_plot

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'eval')
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.07.10/17.22.29_train_diffuser_actor_sweep_to_dustpan_of_size_mask')
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/image')
    config : str = 'train_diffuser_actor.yaml'
    overrides: List[str] = []




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
        for i in range(1):
            batch = next(it)

            batch = dict_apply(batch, lambda x: x.to(device))

            pred = policy.predict_action(batch['obs'])['rlbench_action'].cpu().detach()

            batch = dict_apply(batch, lambda x: x.cpu().detach())

            print("Pred: ", pred[...,7])
            print("GT: ", batch['action']['gt_trajectory'][...,7])

            img = create_obs_state_plot(batch['obs'], gt_action=batch['action']['gt_trajectory'], pred_action=pred, downsample=1, use_mask=False, lowdim = False, quaternion_format='xyzw')
            save_image(torch.tensor(img).float() / 255, os.path.join(save_path, f'pred_{i}.png'))