if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
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
    save_path : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', os.path.basename(__file__).replace('.py', ''))
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'outputs', '2024.06.29/15.39.38_train_diffuser_actor_open_drawer_image_3DDA/')
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/peract/raw')
    config : str = 'train_diffuser_actor.yaml'
    overrides: list = ['task=open_drawer_image_3DDA' , 'num_episodes=1']
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

if __name__ == '__main__':
    args = Arguments().parse_args()


    cfg = load_config(args.config)

    task_str = cfg.task.dataset.task_name
    save_path = args.save_path
    data_root = args.data_root
    hydra_path = args.hydra_path

    os.makedirs(save_path, exist_ok=True)

    runner : RLBenchRunner = hydra.utils.instantiate(cfg.task.env_runner, render_image_size=args.render_image_size, output_dir="")
    
    obs_config = create_obs_config(image_size=cfg.task.image_size, apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
    demos = get_stored_demos(amount=1, dataset_root=data_root, task_name=task_str, variation_number=0, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)
    dataset = load_dataset(cfg)
    policy = load_model(hydra_path, cfg)
    policy = policy.to("cuda")
    policy.eval()
    actioner = Actioner(policy)

    log_data = _evaluate_task_on_demos(env_args=runner.env_args,
                            task_str=runner.task_str,
                            demos=demos[:1],
                            max_steps=3,
                            actioner=actioner,
                            max_rrt_tries=1,
                            demo_tries=1,
                            n_visualize=1,
                            verbose=True)

    if len(log_data['rgbs']) > 0:
        rgbs = log_data['rgbs'][0].transpose(0, 2, 3, 1)
        write_video(rgbs, os.path.join(save_path, "rgbs.mp4"), fps=30)
    else:
        print("No rgbs")

    if len(log_data['obs_state']) > 0:
        obs_state = log_data['obs_state'][0].transpose(0, 2, 3, 1)
        write_video(obs_state, os.path.join(save_path, "obs_state.mp4"), fps=1)
    else:
        print("No obs")

    if len(log_data['mask']) > 0:
        mask = log_data['mask'][0].transpose(0, 2, 3, 1)
        write_video(mask, os.path.join(save_path, "mask.mp4"), fps=1)
    else:
        print("No mask")
    