if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.policy.diffuser_actor import DiffuserActor
from diffusion_policy.dataset.rlbench_next_best_pose_dataset import RLBenchNextBestPoseDataset
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.common.pytorch_util import dict_apply, print_dict
import tap
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from pathlib import Path
import json
from typing import List

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    task: str = 'stack_blocks',
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'eval')
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.07.04/17.46.17_train_diffuser_actor_stack_blocks')
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
    print(args.overrides)
    overrides = load_overrides(args.hydra_path, args.overrides)

    print("Overrides: ", overrides)
    
    cfg = load_config(args.config, overrides)

    hydra_path = args.hydra_path
    print("Config loaded")

    dataset : RLBenchNextBestPoseDataset = load_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Dataset loaded")


    policy : DiffuserActor= load_model(hydra_path, cfg)
    policy = policy.to("cuda")
    policy.eval()
    print("Model loaded")

    batch = next(iter(dataloader))

    print_dict(batch)

    batch = dict_apply(batch, lambda x: x.cuda())

    print("Predicting action")
    output = policy.predict_action(batch['obs'])



