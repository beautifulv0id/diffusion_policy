if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.dataset.rlbench_dataset import RLBenchDataset
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from diffusion_policy.env_runner.rlbench_utils import _evaluate_task_on_demos
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from diffusion_policy.common.logger_utils import write_video
import tap
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from pathlib import Path
import json


class Arguments(tap.Tap):
    task: str = 'open_drawer',
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'eval')
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.06.29/15.39.33_train_diffuser_actor_open_drawer_image_mask_3DDA')
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/peract/raw')
    config : str = 'train_diffuser_actor.yaml'
    overrides: list = []
    render_image_size: tuple = (256, 256)

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

def load_overrides(hydra_path):
    overrides = OmegaConf.load(os.path.join(hydra_path, ".hydra", "overrides.yaml"))
    return overrides

if __name__ == '__main__':
    args = Arguments().parse_args()

    if len(args.overrides) > 0:
        overrides = args.overrides
    else:
        overrides = load_overrides(args.hydra_path)

    cfg = load_config(args.config, overrides)

    task_str = cfg.task.dataset.task_name
    data_root = args.data_root
    hydra_path = args.hydra_path
    save_path = os.path.join(args.save_root, task_str, hydra_path.split('/')[-1])

    os.makedirs(save_path, exist_ok=True)

    runner : RLBenchRunner = hydra.utils.instantiate(cfg.task.env_runner, render_image_size=args.render_image_size, output_dir="")
    env = RLBenchEnv(**runner.env_args)

    dataset : RLBenchDataset = load_dataset(cfg)

    obs_config = create_obs_config(image_size=env.image_size, apply_cameras=env.apply_cameras, apply_pc=env.apply_pc, apply_mask=env.apply_mask, apply_rgb=env.apply_rgb, apply_depth=env.apply_depth)
    demos = get_stored_demos(amount=-1, dataset_root=data_root, task_name=task_str, variation_number=0, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)
    
    policy = load_model(hydra_path, cfg)
    policy = policy.to("cuda")
    policy.eval()
    actioner = Actioner(policy)

    print(f"Hydra path: {hydra_path}")
    print(f"Task: {task_str}")
    print(f"Num demos: {len(demos)}")


    log_data = _evaluate_task_on_demos(env_args=runner.env_args,
                            task_str=runner.task_str,
                            demos=demos,
                            max_steps=3,
                            actioner=actioner,
                            max_rrt_tries=1,
                            demo_tries=1,
                            n_visualize=len(demos),
                            verbose=True,
                            n_procs_max=1)
    
    success_rate = log_data['success_rate']
    print(f"Success rate: {success_rate}")

    with open(os.path.join(save_path, "metrics.json"), 'w') as f:
        json.dump({
            "success_rate": success_rate
        }, f)

    if len(log_data['rgbs']) > 0:
        rgbs = log_data['rgbs'][0].transpose(0, 2, 3, 1)
        for i, rgbs in enumerate(log_data['rgbs']):
            rgbs = rgbs.transpose(0, 2, 3, 1)
            write_video(rgbs, os.path.join(save_path, f"rgbs_{i}.mp4"), fps=30)
    else:
        print("No rgbs")

    if len(log_data['obs_state']) > 0:
        for i, obs_state in enumerate(log_data['obs_state']):
            obs_state = obs_state.transpose(0, 2, 3, 1)
            write_video(obs_state, os.path.join(save_path, f"obs_state_{i}.mp4"), fps=30)
    else:
        print("No obs")

    if len(log_data['mask']) > 0:
        for i, masks in enumerate(log_data['mask']):
            masks = masks.transpose(0, 2, 3, 1)
            write_video(masks, os.path.join(save_path, f"mask_{i}.mp4"), fps=30)
    else:
        print("No mask")
    