import time
import math

if __name__ == "__main__":
    import multiprocessing
    import matplotlib
    import sys
    import os
    import pathlib
    multiprocessing.set_start_method('spawn')
    matplotlib.use('Agg')

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.json_logger import JsonLogger
import omegaconf

from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.env.rlbench.rlbench_utils import Actioner
from diffusion_policy.common.rlbench_util import (
    load_instructions,
    round_floats,
    get_gripper_loc_bounds
)
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # open previous config
        train_config = omegaconf.OmegaConf.load(cfg.model_to_eval_path+'/config_raw.yaml')

        OmegaConf.resolve(cfg)

        # dump current config to yaml file:
        evaluation_folder_name = "EVALUATE_" + cfg.evaluation_folder_name + '_tasks_' + ''.join(cfg.tasks) + '_variations_' + str(cfg.variations[-1])
        evaluation_dir_path = cfg.model_to_eval_path + '/' + evaluation_folder_name + '/'
        os.makedirs(evaluation_dir_path, exist_ok=True)
        with open(evaluation_dir_path + "/config_raw_eval.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        with open(evaluation_dir_path + "/config_raw_corr_train.yaml", "w") as f:
            OmegaConf.save(train_config, f)
        self.evaluation_dir_path = evaluation_dir_path

        # this evaluates the arguments!
        train_config = OmegaConf.create(OmegaConf.to_yaml(train_config, resolve=True))

        # naming: eval_cfg is the config for the evaluation, cfg is the config that comes from the orignal training run
        eval_cfg = copy.deepcopy(cfg)
        cfg = train_config

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.eval_cfg = eval_cfg

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model)

    def rollout(self, wandb_run=None, post_train=False):
        cfg = copy.deepcopy(self.cfg)
        eval_cfg = copy.deepcopy(self.eval_cfg)

        # for now the device is the same as used during training - maybe we want to change that?!
        device = torch.device(cfg.training.device)
        # now load the model and the actual checkpoint!
        self.load_checkpoint(path=eval_cfg.model_to_eval_path + '/checkpoints/' + eval_cfg.checkpoint, exclude_keys='optimizer')
        if (eval_cfg.use_ema_model):
            policy = self.ema_model
        else:
            policy = self.model
        policy.to(device).eval()

        # Load RLBench environment
        env = RLBenchEnv(
            data_path=eval_cfg.eval_data_dir,
            image_size=cfg.image_size, #i.e. use the same image size as during train,... [int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(eval_cfg.headless),
            apply_cameras=cfg.cameras, #args.cameras,
            collision_checking=bool(eval_cfg.collision_checking)
        )

        instruction = load_instructions(eval_cfg.eval_instruction_dir)
        if instruction is None:
            raise NotImplementedError()

        action_dim = cfg.env_runner.action_dim
        actioner = Actioner(policy, instructions=instruction, action_dim=action_dim)

        # max_eps_dict = load_episodes()["max_episode_length"] # not functional now
        max_eps_dict = eval_cfg.max_steps
        task_success_rates = {}

        for task_str in eval_cfg.tasks:
            var_success_rates = env.evaluate_task_on_multiple_variations(
                task_str,
                # max_steps=(
                #     max_eps_dict[task_str] if args.max_steps == -1
                #     else args.max_steps
                # ),
                max_steps=eval_cfg.max_steps,
                num_variations=eval_cfg.variations[-1] + 1,
                num_demos=eval_cfg.num_episodes,
                actioner=actioner,
                max_tries=eval_cfg.max_tries,
                dense_interpolation=bool(eval_cfg.dense_interpolation),
                interpolation_length=eval_cfg.interpolation_length,
                verbose=bool(eval_cfg.verbose),
                num_history=cfg.n_obs_steps #TODO - double check if this is right to be set to the observation steps of the policy but I think it makes sense - was before: args.num_history
            )
            print()
            print(
                f"{task_str} variation success rates:",
                round_floats(var_success_rates)
            )
            print(
                f"{task_str} mean success rate:",
                round_floats(var_success_rates["mean"])
            )

            task_success_rates[task_str] = var_success_rates
            with open(self.evaluation_dir_path + 'eval_log.json', "w") as f:
                json.dump(round_floats(task_success_rates), f, indent=4)

        print ("Finished the evaluation!")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainingWorkspace(cfg)
    workspace.rollout()
    print ("ROLLOUT COMPLETED")

if __name__ == "__main__":
    main()