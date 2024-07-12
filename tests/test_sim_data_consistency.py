if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

import os
import torch
from rlbench.utils import get_stored_demos
from diffusion_policy.env_runner.rlbench_runner import RLBenchRunner
from diffusion_policy.dataset.rlbench_zarr_dataset import RLBenchNextBestPoseDataset
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS
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
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class, Actioner, Mover, get_actions_from_demo
from diffusion_policy.common.rlbench_util import extract_obs, create_obs_config
from rlbench.task_environment import TaskEnvironment
from rlbench.demo import Demo
import torch.nn.functional as F
from diffusion_policy.common.pytorch_util import dict_apply
from PIL import Image

OmegaConf.register_new_resolver("eval", eval, replace=True)

class Arguments(tap.Tap):
    save_root : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'tests', os.path.basename(__file__).replace('.py', ''))
    data_root = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/image')
    config : str = 'train_diffuser_actor.yaml'
    overrides: List[str] = []
    render_image_size: tuple = (256, 256)
    n_demos: int = 1


class ReplayPolicy:
    def __init__(self, demo):
        self._actions = get_actions_from_demo(demo)
        self.idx = 0
        self.n_obs_steps = 1

    def predict_action(self, obs):
        action = self._actions[self.idx % len(self._actions)]
        self.idx += 1
        return {
            'rlbench_action' : 
                action.unsqueeze(0)
        }
    
    def eval(self):
        pass

    def parameters(self):
        return iter([torch.empty(0)])
    
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
    overrides =  args.overrides

    print("Overrides: ", overrides)
    
    cfg = load_config(args.config, overrides=overrides)

    task_str = cfg.task.dataset.task_name
    save_path = os.path.join(args.save_root, cfg.task.name)

    os.makedirs(save_path, exist_ok=True)

    runner : RLBenchRunner = hydra.utils.instantiate(cfg.task.env_runner, render_image_size=args.render_image_size, output_dir="")
    env = RLBenchEnv(**runner.env_args)

    dataset : RLBenchNextBestPoseDataset = load_dataset(cfg)

    obs_config = create_obs_config(image_size=env.image_size, apply_cameras=[], apply_pc=False, apply_mask=False, apply_rgb=False, apply_depth=False)
    demo = get_stored_demos(amount=1, dataset_root=args.data_root, task_name=task_str, variation_number=0, from_episode_number=0, image_paths=False, random_selection=False, obs_config=obs_config)[0]


    actioner = Actioner(
        policy=ReplayPolicy(demo),
        action_dim=8,
    )

    log_data = _evaluate_task_on_demos(env_args=runner.env_args,
                            task_str=runner.task_str,
                            demos=[demo],
                            max_steps=3,
                            actioner=actioner,
                            max_rrt_tries=1,
                            demo_tries=1,
                            n_visualize=1,
                            verbose=True,
                            return_model_obs=True,)
    
    keypoint_obs = log_data['model_obs'][0]

    for i, demo_obs in enumerate(keypoint_obs):
        demo_obs = dict_apply(demo_obs, lambda x: x[0])
        dataset_obs = dataset[i]['obs']
        for key, data in dataset_obs.items():
            demo_data = demo_obs[key]
            print(key)
            print(data.shape, demo_data.shape)
            print(data.dtype, demo_data.dtype)
            print(data.max(), demo_data.max())
            if key.endswith('mask'):
                for j, (img, camera) in enumerate(zip(data, env.apply_cameras)):
                    img = img.permute(1, 2, 0) * 255
                    img = Image.fromarray(img.squeeze().cpu().numpy().astype('uint8'))
                    img.save(os.path.join(save_path, f'{i}_{key}_{camera}_dataset.png'))
            elif key.endswith('rgb'):
                for j, (img, camera) in enumerate(zip(data, env.apply_cameras)):
                    img = (img.permute(1, 2, 0) * 255).type(torch.uint8)
                    img = Image.fromarray(img.cpu().numpy())
                    img.save(os.path.join(save_path, f'{i}_{key}_{camera}_dataset.png'))
        for key, data in demo_obs.items():
            if key.endswith('mask'):
                for j, (img, camera) in enumerate(zip(data, env.apply_cameras)):
                    img = img.permute(1, 2, 0) * 255
                    img = Image.fromarray(img.squeeze().cpu().numpy().astype('uint8'))
                    img.save(os.path.join(save_path, f'{i}_{key}_{camera}_demo.png'))
            elif key.endswith('rgb'):
                for j, (img, camera) in enumerate(zip(data, env.apply_cameras)):
                    img = (img.permute(1, 2, 0) * 255).type(torch.uint8)
                    img = Image.fromarray(img.cpu().numpy())
                    img.save(os.path.join(save_path, f'{i}_{key}_{camera}_demo.png'))

