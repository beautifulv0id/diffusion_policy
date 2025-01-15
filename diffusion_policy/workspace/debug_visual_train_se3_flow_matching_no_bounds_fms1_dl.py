import time
import math

if __name__ == "__main__":
    import multiprocessing
    import matplotlib
    import sys
    import os
    import pathlib
    multiprocessing.set_start_method('spawn')
    matplotlib.use('TkAgg')

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
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.se3_flow_matching import SE3FlowMatching
from diffusion_policy.common.rlbench_util import create_obs_state_plot
from torchvision.utils import make_grid

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # dump current config to yaml file:
        with open(self.output_dir + "/config_raw.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model : SE3FlowMatching = hydra.utils.instantiate(cfg.policy)
        self.ema_model: SE3FlowMatching = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure data
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()
        # configure validation data
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        # self.model.set_normalizer(normalizer)
        # if cfg.training.use_ema:
        #     self.ema_model.set_normalizer(normalizer)

        std = 4*torch.Tensor([0.5, 0.5, 0.5, math.pi, math.pi, math.pi])[None, ...]
        mean = torch.Tensor([0, 0, 0, 0, 0, 0])[None, ...]
        self.model.flow.set_mean_std(mean, std)

        # self.model.set_mean_std(*dataset.get_mean_std(
        #     relative_to_gripper=cfg.policy.relative,
        #     quaternion_format=cfg.policy.quaternion_format)
        #     )

        # configure env
        # env_runner: BaseImageRunner
        if 'env_runner' in cfg.task.keys():
            # do this in order to avoid loading the data again
            if ("real_robot" in cfg.task.keys()):
                if (cfg.task.real_robot):
                    env_runner = hydra.utils.instantiate(
                        cfg.task.env_runner)
                    env_runner.initialize(dataset)
            else:
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir)
        else:
            env_runner = None


        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        dtype = self.model.dtype

        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # if normalizer is not None:
        #     normalizer_to(normalizer, device, dtype)

        # save batch for sampling
        train_sampling_batch = None
        val_sampling_batch = None



        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            with tqdm.tqdm(range(self.epoch, cfg.training.num_epochs), desc="Training",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as gepoch:

                for local_epoch_idx in gepoch:
                    step_log = dict()
                    # ========= train for this epoch ==========
                    train_losses = list()
                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                                            
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                                
                            batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True))


                            train_sampling_batch = dict_apply(train_sampling_batch, lambda x: x[:cfg.training.visualize_batch_size])
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, dtype, non_blocking=True))
                            policy = self.model
                            pred = policy.predict_action(batch['obs'])
                            obs = train_sampling_batch['obs']
                            gt_action = train_sampling_batch['action']['gt_trajectory']
                            pred_action = pred['trajectory'].cpu().detach()
                            imgs = create_obs_state_plot(obs=obs, gt_action=gt_action, pred_action=pred_action, quaternion_format=policy._quaternion_format, lowdim=cfg.task.type == 'lowdim')
                            img = make_grid(torch.from_numpy(imgs).float() / 255)
                            import matplotlib.pyplot as plt
                            plt.imshow(np.transpose(np.transpose(img), (1, 0, 2)))


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainingWorkspace(cfg)
    # only has run mode!
    workspace.run()

if __name__ == "__main__":
    main()