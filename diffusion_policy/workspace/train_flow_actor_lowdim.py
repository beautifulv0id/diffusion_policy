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
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.diffuser_actor import DiffuserActor
from diffusion_policy.common.rlbench_util import create_obs_state_plot
from torchvision.utils import make_grid

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model : DiffuserActor = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1
                self.global_step += 1

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

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                                       len(train_dataloader) * cfg.training.num_epochs) \
                               // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )

        # # configure ema
        # ema: EMAModel = None
        # if cfg.training.use_ema:
        #     ema = hydra.utils.instantiate(
        #         cfg.ema,
        #         model=self.ema_model)

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

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        dtype = self.model.dtype
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # if normalizer is not None:
        #     normalizer_to(normalizer, device, dtype)

        # save batch for sampling
        train_sampling_batch = None
        val_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.training.visualize_every = 1
            image = wandb.Image(dataset.get_data_visualization(), caption="Dataset")
            wandb_run.log({"dataset": image}, step=self.global_step)


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            with tqdm.tqdm(range(self.epoch, cfg.training.num_epochs), desc="Training",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as gepoch:

                for local_epoch_idx in gepoch:
                    step_log = dict()
                    # ========= train for this epoch ==========
                    # if cfg.training.freeze_encoder:
                    #     self.model.obs_encoder.eval()
                    #     self.model.obs_encoder.requires_grad_(False)

                    train_losses = list()
                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                                            
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                                
                            batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True))

                            # compute loss
                            raw_loss = self.model.compute_loss(batch)

                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()

                            # update ema
                            # if cfg.training.use_ema:
                            #     ema.step(self.model)

                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)


                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                    and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                    # ========= eval for this epoch ==========
                    policy = self.model
                    # if cfg.training.use_ema:
                    #     policy = self.ema_model
                    policy.eval()

                    # run rollout (TASK SATISFACTION)
                    if (self.epoch % cfg.training.rollout_every) == 0 \
                            and self.epoch > 0:
                        dataset.empty_cache() # empty cache before running
                        val_dataset.empty_cache()
                        runner_log = env_runner.run(policy, dataset.demos, mode="train")
                        runner_log.update(
                            env_runner.run(policy, val_dataset.demos, mode="eval")
                        )
                        # log all
                        step_log.update(runner_log)

                    # run validation
                    if (self.epoch % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    # batch = format_batch(batch)
                                    batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True))
                                    if val_sampling_batch is None:
                                        val_sampling_batch = batch

                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                            and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss


                    ## Run Experiment related Validation ## #TODO: as far as I see, this currently has no effect!
                    if (self.epoch % cfg.training.model_evaluation_every) == 0:
                        evaluation_log = self.model.evaluate(val_sampling_batch, validation=True)
                        # log all
                        step_log.update(evaluation_log)

                    # sample on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))

                            eval_log = policy.evaluate(batch)
                            # log all
                            step_log.update(eval_log)

                    if (self.epoch % cfg.training.visualize_every) == 0:
                        with torch.no_grad():
                            train_sampling_batch = dict_apply(train_sampling_batch, lambda x: x[:cfg.training.visualize_batch_size])
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            pred = policy.predict_action(batch['obs'])
                            obs = train_sampling_batch['obs']
                            gt_action = train_sampling_batch['action']['gt_trajectory']
                            pred_action = pred['rlbench_action'].cpu().detach()
                            imgs = create_obs_state_plot(obs=obs, gt_action=gt_action, pred_action=pred_action, quaternion_format=policy._quaternion_format, lowdim=cfg.task.type == 'lowdim')
                            img = make_grid(torch.from_numpy(imgs).float() / 255)
                            image = wandb.Image(img, caption="Prediction vs Ground Truth")
                            wandb_run.log({"prediction_vs_gt": image}, step=self.global_step)


                    # checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    # ========= eval end for this epoch ==========
                    policy.train()

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
                    gepoch.set_postfix(train_loss=train_loss, refresh=False)
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainingWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()