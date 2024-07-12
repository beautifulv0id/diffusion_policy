if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.dataset.rlbench_zarr_dataset import RLBenchNextBestPoseDataset
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
from typing import List
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class, Actioner, Mover
from diffusion_policy.common.rlbench_util import extract_obs
from rlbench.task_environment import TaskEnvironment
from rlbench.demo import Demo
import torch.nn.functional as F
from diffusion_policy.common.pytorch_util import dict_apply
from PIL import Image

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', os.path.basename(__file__).replace('.py', ''))
    hydra_path: str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/outputs/2024.07.10/17.22.29_train_diffuser_actor_sweep_to_dustpan_of_size_mask')
    config : str = 'train_diffuser_actor.yaml'
    overrides: List[str] = []
    render_image_size: tuple = (256, 256)
    n_demos: int = 1

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
    hydra_path = args.hydra_path
    save_path = os.path.join(args.save_root, task_str, hydra_path.split('/')[-1])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32

    os.makedirs(save_path, exist_ok=True)

    runner : RLBenchRunner = hydra.utils.instantiate(cfg.task.env_runner, render_image_size=args.render_image_size, output_dir="")
    env = RLBenchEnv(**runner.env_args)

    dataset : RLBenchNextBestPoseDataset = load_dataset(cfg)

    demos = dataset.demos[:args.n_demos]

    policy = load_model(hydra_path, cfg)
    policy = policy.to(device)
    policy.eval()
    actioner = Actioner(policy)

    print(f"Hydra path: {hydra_path}")
    print(f"Task: {task_str}")
    print(f"Num demos: {len(demos)}")

    n_obs_steps = cfg.n_obs_steps

    task_type = task_file_to_task_class(task_str)
    task : TaskEnvironment = env.env.get_task(task_type)
    task.set_variation(0)


    first_obs = []
    for i, demo in enumerate(demos):
        print(f"Demo: {i}")
        rgbs = torch.Tensor([])
        pcds = torch.Tensor([])
        masks = torch.Tensor([])
        low_dim_pcds = torch.Tensor([])
        keypoint_poses = torch.Tensor([])
        grippers = torch.Tensor([])
        low_dim_states = torch.Tensor([])

        descriptions, observation = task.reset_to_demo(demo)
        obs_dict = extract_obs(observation, cameras=env.apply_cameras, use_rgb=env.apply_rgb, use_pcd=env.apply_pc, use_mask=env.apply_mask, use_pose=env.apply_poses, use_low_dim_state=True, use_low_dim_pcd=env.apply_low_dim_pcd)

        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

        if env.apply_rgb:
            rgb = obs_dict['rgb']
            rgbs = torch.cat([rgbs, rgb], dim=0)
        if env.apply_pc:
            pcd = obs_dict['pcd']
            pcds = torch.cat([pcds, pcd], dim=0)
        if env.apply_low_dim_pcd:
            low_dim_pcd = obs_dict['low_dim_pcd']
            low_dim_pcds = torch.cat([low_dim_pcds, low_dim_pcd], dim=0)
        if env.apply_poses:
            keypoint_pose = obs_dict['keypoint_poses']
            keypoint_poses = torch.cat([keypoint_poses, keypoint_pose], dim=0)
        if env.apply_mask:
            mask = obs_dict['mask']
            masks = torch.cat([masks, mask], dim=0)

        gripper = obs_dict['curr_gripper']
        grippers = torch.cat([grippers, gripper], dim=0)
        low_dim_state = obs_dict['low_dim_state']
        low_dim_states = torch.cat([low_dim_states, low_dim_state], dim=0)
    
        def pad_input(input : torch.Tensor, npad):
            sh_in = input.shape
            input = input[-n_obs_steps:].unsqueeze(0)
            input = input.reshape(input.shape[:2] + (-1,))
            input = F.pad(
                input, (0, 0, npad, 0), mode='replicate'
            )
            input = input.view((1, n_obs_steps, ) + sh_in[1:])
            return input

        # Prepare proprioception history
        npad = n_obs_steps - grippers[-n_obs_steps:].shape[0]
        obs_dict = dict()
        obs_dict["curr_gripper"] = pad_input(grippers, npad)
        grippers = grippers[-n_obs_steps:]
        obs_dict["low_dim_state"] = pad_input(low_dim_states, npad)
        low_dim_states = low_dim_states[-n_obs_steps:]

        if env.apply_rgb:
            obs_dict["rgb"] = rgbs[-1:]
            rgbs = rgbs[-n_obs_steps:]
        if env.apply_pc:
            obs_dict["pcd"]  = pcds[-1:]
            pcds = pcds[-n_obs_steps:]
        if env.apply_low_dim_pcd:
            obs_dict["low_dim_pcd"] = low_dim_pcds[-1:]
            low_dim_pcds = low_dim_pcds[-n_obs_steps:]
        if env.apply_poses:
            obs_dict["keypoint_poses"] = keypoint_poses[-1:]
            keypoint_poses = keypoint_poses[-n_obs_steps:]
        if env.apply_mask:
            obs_dict["mask"] = masks[-1:].bool()
            masks = masks[-n_obs_steps:]

        obs_dict = dict_apply(obs_dict, lambda x: x.type(dtype).to(device))
        # out = actioner.predict(obs_dict)
        # trajectory = out['rlbench_action']

        obs_dict = dict_apply(obs_dict, lambda x: x.type(dtype).to(device).squeeze(0))
        first_obs.append(obs_dict)

    for i, demo_obs in enumerate(first_obs):
        dataset_obs = dataset[0]['obs']
        for key, data in dataset_obs.items():
            demo_data = demo_obs[key]
            print(key)
            print(data.shape, demo_data.shape)
            print(data.dtype, demo_data.dtype)
            print(data.max(), demo_data.max())
            if key.endswith('mask'):
                for j, img in enumerate(data):
                    img = img.permute(1, 2, 0) * 255
                    img = Image.fromarray(img.squeeze().cpu().numpy().astype('uint8'))
                    img.save(os.path.join(save_path, f'{j}_{key}_dataset.png'))
            elif key.endswith('rgb'):
                for j, img in enumerate(data):
                    img = (img.permute(1, 2, 0) * 255).type(torch.uint8)
                    img = Image.fromarray(img.cpu().numpy())
                    img.save(os.path.join(save_path, f'{j}_{key}_dataset.png'))
        for key, data in demo_obs.items():
            if key.endswith('mask'):
                for j, img in enumerate(data):
                    img = img.permute(1, 2, 0) * 255
                    img = Image.fromarray(img.squeeze().cpu().numpy().astype('uint8'))
                    img.save(os.path.join(save_path, f'{j}_{key}_demo.png'))
            elif key.endswith('rgb'):
                for j, img in enumerate(data):
                    img = (img.permute(1, 2, 0) * 255).type(torch.uint8)
                    img = Image.fromarray(img.cpu().numpy())
                    img.save(os.path.join(save_path, f'{j}_{key}_demo.png'))

