import time

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
import math
import hydra
import torch
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
from diffusion_policy.common.rlbench_util import load_instructions
from diffusion_policy.model.common.trajectory_criterion import TrajectoryCriterion
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pickle

OmegaConf.register_new_resolver("eval", eval, replace=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None, cfg_unresolved=None):
        super().__init__(cfg, output_dir=output_dir)

        # dump current config to yaml file:
        if dist.get_rank() == 0:
            # dump current config to yaml file:
            if not os.path.exists(self.output_dir + "/config_raw.yaml"):
                with open(self.output_dir + "/config_raw.yaml", "w") as f:
                    if cfg_unresolved is not None:
                        OmegaConf.save(cfg_unresolved, f)
                    else:
                        OmegaConf.save(cfg, f)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def get_dataset(self, cfg):
        instructions = load_instructions(
            cfg.instructions,
            tasks=cfg.tasks,
            variations=cfg.variations
        )
        taskvar = [
                (task, var)
                for task, var_instr in instructions.items()
                for var in var_instr.keys()
            ]
        dataset = hydra.utils.instantiate(cfg.dataset, taskvar=taskvar, instructions=instructions)
        return dataset

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
        dataset = self.get_dataset(cfg)
        g = torch.Generator()
        g.manual_seed(0)
        train_dataloader = DataLoader(dataset, **cfg.dataloader, sampler=DistributedSampler(dataset), generator=g)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader, sampler=DistributedSampler(val_dataset), generator=g)
        std = torch.Tensor([2.0, 2.0, 2.0, math.pi, math.pi, math.pi])[None, ...]
        mean = torch.Tensor([0, 0, 0, 0, 0, 0])[None, ...]
        self.model.flow.set_mean_std(mean, std)

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

        # configure logging
        if dist.get_rank() == 0:
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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.training.visualize_every = 1
            cfg.env_runner.max_episodes = 1

        # # configure ema
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = 'cuda:{}'.format(torch.cuda.current_device())
        self.model = self.model.to(device)
        dtype = self.model.dtype
        rank = dist.get_rank()
        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()
        self.model = DistributedDataParallel(self.model, device_ids=[device_id],
                                        broadcast_buffers=False, find_unused_parameters=True
        )

        if self.ema_model is not None:
            self.ema_model.to(device)

        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        val_sampling_batch = None

        criterion = TrajectoryCriterion(quaternion_format=cfg.policy.quaternion_format)

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
                                
                            batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True) if isinstance(x, torch.Tensor) else x)

                            # compute loss
                            raw_loss = self.model(
                                gt_trajectory=batch['action']['gt_trajectory'],
                                rgb_obs=batch['obs'].get('rgb', None),
                                pcd_obs=batch['obs']['pcd'],
                                instruction=batch['obs'].get('instruction', None),
                                curr_gripper=batch['obs']['curr_gripper'],
                                run_inference=False,
                                feature_obs=batch['obs'].get('clip_features', None)
                            )
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()

                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model.module)

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
                                if dist.get_rank() == 0:
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
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run validation
                    if ((self.epoch + 1) % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    # batch = format_batch(batch)
                                    batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                                    if val_sampling_batch is None:
                                        val_sampling_batch = batch

                                    loss = policy(
                                        gt_trajectory=batch['action']['gt_trajectory'],
                                        rgb_obs=batch['obs'].get('rgb', None),
                                        pcd_obs=batch['obs']['pcd'],
                                        instruction=batch['obs'].get('instruction', None),
                                        curr_gripper=batch['obs']['curr_gripper'],
                                        run_inference=False,
                                        feature_obs=batch['obs'].get('clip_features', None)
                                    )
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                            and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                values = self.synchronize_between_processes({'val_loss': torch.tensor(val_losses)})
                                val_loss = torch.mean(values['val_loss']).item()
                                if dist.get_rank() == 0:
                                    step_log['val_loss'] = val_loss

                    if ((self.epoch + 1) % cfg.training.model_evaluation_every) == 0:
                        with torch.no_grad():
                            values = {}
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True) if isinstance(x, torch.Tensor) else x)
                                    pred_act = policy(
                                        gt_trajectory=None,
                                        rgb_obs=batch['obs'].get('rgb', None),
                                        pcd_obs=batch['obs']['pcd'],
                                        instruction=batch['obs'].get('instruction', None),
                                        curr_gripper=batch['obs']['curr_gripper'],
                                        run_inference=True,
                                        feature_obs=batch['obs'].get('clip_features', None)
                                    )
                                    # log all
                                    evaluation_log = criterion.compute_metrics(pred_act, batch, validation=True)
                                    
                                    # gather per-task metrics
                                    for key, val in evaluation_log.items():
                                        if key not in values:
                                            values[key] = torch.Tensor([]).to(device)
                                        values[key] = torch.cat([values[key], val.unsqueeze(0)])

                                    if (cfg.training.max_val_steps is not None) \
                                            and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            
                            values = self.synchronize_between_processes(values)
                            values = {k: v.mean().item() for k, v in values.items()}
                            if dist.get_rank() == 0:
                                step_log.update(values)

                    # sample on a training batch
                    if ((self.epoch + 1) % cfg.training.sample_every) == 0 and dist.get_rank() == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, dtype, non_blocking=True) if isinstance(x, torch.Tensor) else x)

                            pred_act = policy(
                                gt_trajectory=None,
                                rgb_obs=batch['obs'].get('rgb', None),
                                pcd_obs=batch['obs']['pcd'],
                                instruction=batch['obs'].get('instruction', None),
                                curr_gripper=batch['obs']['curr_gripper'],
                                run_inference=True,
                                feature_obs=batch['obs'].get('clip_features', None)
                            )
                            eval_log = criterion.compute_metrics(pred_act, batch)
                            # log all
                            step_log.update(eval_log)

                    # checkpoint
                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    if ((self.epoch + 1) % cfg.training.save_milestone_every) == 0 and dist.get_rank() == 0:
                        self.save_checkpoint(tag="epoch="+str(self.epoch).zfill(4))

                    if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and dist.get_rank() == 0:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

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
                    if dist.get_rank() == 0:
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
                    gepoch.set_postfix(train_loss=train_loss, refresh=False)


    def synchronize_between_processes(self, a_dict):
        all_dicts = all_gather(a_dict)

        if not is_dist_avail_and_initialized() or dist.get_rank() == 0:
            merged = {}
            for key in all_dicts[0].keys():
                device = all_dicts[0][key].device
                merged[key] = torch.cat([
                    p[key].to(device) for p in all_dicts
                    if key in p
                ])
            a_dict = merged
        return a_dict

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty(
            (max_size,), dtype=torch.uint8, device="cuda"
        ))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
