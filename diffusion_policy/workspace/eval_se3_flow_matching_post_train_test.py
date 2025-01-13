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
from diffusion_policy.workspace.diffusion_workspace import BaseWorkspace
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.se3_flow_matching import SE3FlowMatching
from diffusion_policy.common.rlbench_util import create_obs_state_plot
from torchvision.utils import make_grid
import omegaconf

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # open previous config
        train_config = omegaconf.OmegaConf.load(cfg.model_to_eval_path+'/config_raw.yaml')

        # dump current config to yaml file:
        with open(self.output_dir + "/config_raw_eval.yaml", "w") as f:
            OmegaConf.save(cfg, f)
        with open(self.output_dir + "/config_raw_corr_train.yaml", "w") as f:
            OmegaConf.save(train_config, f)

        # this evaluates the arguments!
        train_config = OmegaConf.create(OmegaConf.to_yaml(train_config, resolve=True))

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
        self.model : SE3FlowMatching = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def rollout(self, wandb_run=None, post_train=False):
        cfg = copy.deepcopy(self.cfg)
        eval_cfg = copy.deepcopy(self.eval_cfg)

        # get all checkpoints!
        filepath = self.output_dir + '/checkpoints'
        # mod_filepath = '/home/funk/Code_TUD/3d_repr_learning/diffusion_policy/data/outputs/cluster_trains/18.22.16_train_se3_flow_matching_no_bounds_fms1_dl_open_drawer_image'  + '/checkpoints'
        mod_filepath =  eval_cfg.model_to_eval_path + '/checkpoints'
        filepath = mod_filepath
        # now list all the checkpoints:
        checkpoint_list = os.listdir(filepath)

        # now go through all of them:
        all_checkpoints = []
        checkpoint_epoch = []
        for checkpoint in checkpoint_list:
            if (checkpoint[-5:]==".ckpt" and checkpoint[:6]=="epoch="):
                if (post_train):
                    # if after training - only additionally roll out the checkpoints that have not been rolled out before
                    if (int(checkpoint[6:10]))%cfg.training.rollout_every!=0 or int(checkpoint[6:10])==0:
                        all_checkpoints.append(checkpoint)
                        checkpoint_epoch.append(int(checkpoint[6:10]))
                else:
                    all_checkpoints.append(checkpoint)
                    checkpoint_epoch.append(int(checkpoint[6:10]))


        # now sort them:
        checkpoint_epoch = np.array(checkpoint_epoch)
        sorted_indices = np.argsort(checkpoint_epoch)
        all_checkpoints = np.array(all_checkpoints)[sorted_indices]
        checkpoint_epoch = checkpoint_epoch[sorted_indices]

        if ('epochs_to_eval' in eval_cfg):
            # only evaluate the specified epochs
            checkpoints_to_eval = []
            checkpoint_epoch_to_eval = []
            for j in range(len(all_checkpoints)):
                if checkpoint_epoch[j] in eval_cfg.epochs_to_eval:
                    checkpoints_to_eval.append(all_checkpoints[j])
                    checkpoint_epoch_to_eval.append(checkpoint_epoch[j])
            all_checkpoints = checkpoints_to_eval
            checkpoint_epoch = checkpoint_epoch_to_eval

        log_path = os.path.join(self.output_dir, 'eval_logs.json.txt')
        if wandb_run is None:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(eval_cfg, resolve=True),
                **eval_cfg.logging
            )

        with JsonLogger(log_path) as json_logger:

            for j in range(len(all_checkpoints)):
                if j>0 and checkpoint_epoch[j]==checkpoint_epoch[j-1]:
                    # skip if there are multiple checkpoints for the same epoch
                    continue
                for jj in range(3):
                    # load the current checkpoint
                    print ("Loading checkpoint: ", all_checkpoints[j])
                    # self.load_checkpoint(path=filepath + '/' + all_checkpoints[j])
                    self.load_checkpoint(path=mod_filepath + '/' + all_checkpoints[j], exclude_keys='optimizer')

                    device = torch.device(cfg.training.device)
                    self.model.to(device)
                    policy = self.model
                    policy.eval()

                    # overwrite some configs which might have been set different during train,...
                    if 'n_vis' in eval_cfg.keys():
                        cfg.task.n_train_vis = min(eval_cfg.n_vis,1)
                        cfg.task.n_val_vis = min(eval_cfg.n_vis,1)
                    if 'n_procs_max' in eval_cfg.keys():
                        cfg.task.env_runner.n_procs_max = eval_cfg.n_procs_max

                    env_runner = hydra.utils.instantiate(
                        cfg.task.env_runner,
                        output_dir=self.output_dir)
                    dataset = hydra.utils.instantiate(cfg.task.dataset)
                    val_dataset = dataset.get_test_dataset()

                    with torch.no_grad():
                        env_runner.max_rrt_tries = 10
                        if (jj==0):
                            runner_log = env_runner.run(policy, cfg.policy, dataset.demos, mode="train")
                            runner_log.update(
                                env_runner.run(policy, cfg.policy, val_dataset.demos, mode="eval")
                            )
                        else:
                            runner_log.update(env_runner.run(policy, cfg.policy, dataset.demos, mode="train"))
                            runner_log.update(
                                env_runner.run(policy, cfg.policy, val_dataset.demos, mode="eval"))
                        runner_log['epoch'] = int(checkpoint_epoch[j])
                        # log all
                        wandb_run.log(runner_log)
                        json_logger.log(runner_log)

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