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
from diffusion_policy.policy.se3_flow_matching import SE3FlowMatching
from diffusion_policy.common.rlbench_util import create_obs_state_plot, load_instructions
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from diffusion_policy.common.rotation_utils import normalise_quat
from pytorch3d.transforms import quaternion_to_matrix
from diffusion_policy.common.so3_util import log_map
from omegaconf import OmegaConf
import pickle

OmegaConf.register_new_resolver("eval", eval, replace=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class TrajectoryCriterion:
    def __init__(self, quaternion_format='xyzw'):
        self._quaternion_format = quaternion_format

    def compute_loss(self, pred, gt=None):
        return pred

    def compute_metrics(self, pred_act, pred_act_gr, batch, validation=False):
        log_dict = {}

        gt_trajectory = batch['action']['gt_trajectory']
        gt_act_p = gt_trajectory[..., :3]
        gt_act_r = gt_trajectory[..., 3:7]
        if self._quaternion_format == 'xyzw':
            gt_act_r = gt_act_r[..., (3, 0, 1, 2)]
        gt_act_r = normalise_quat(gt_act_r)
        gt_act_r = quaternion_to_matrix(gt_act_r)
        gt_act_gr = gt_trajectory[..., 7:8]

        pred_act_p = pred_act[..., :3, -1]
        pred_act_r = pred_act[..., :3, :3]

        pos_error = torch.nn.functional.mse_loss(pred_act_p, gt_act_p)

        R_inv_gt = torch.transpose(gt_act_r, -1, -2)
        relative_R = torch.matmul(R_inv_gt, pred_act_r)
        angle_error = log_map(relative_R)
        rot_error = torch.nn.functional.mse_loss(angle_error, torch.zeros_like(angle_error))
        gr_error = torch.nn.functional.l1_loss(pred_act_gr, gt_act_gr)

        prefix = 'val_' if validation else 'train_'
        log_dict[prefix + 'gripper_l1_loss'] = gr_error.item()
        log_dict[prefix + 'position_mse_error'] = pos_error.item()
        log_dict[prefix + 'rotation_mse_error'] = rot_error.item()

        return log_dict


class TrainingWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # dump current config to yaml file:
        if dist.get_rank() == 0:
            with open(self.output_dir + "/config_raw.yaml", "w") as f:
                OmegaConf.save(cfg, f)

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
        self.model.set_mean_std(*dataset.get_mean_std(
            relative_to_gripper=cfg.policy.relative, 
            quaternion_format=cfg.policy.quaternion_format)
            )

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
        ema: SE3FlowMatching = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner = hydra.utils.instantiate(
            cfg.env_runner,
            output_dir=self.output_dir)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model = self.model.to(device)
        dtype = self.model.dtype
        self.model = DistributedDataParallel(self.model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                        broadcast_buffers=False, find_unused_parameters=True
        )

        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        # if normalizer is not None:
        #     normalizer_to(normalizer, device, dtype)

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
                                
                            batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True))

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
                                ema.step(self.model)

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
                    # if cfg.training.use_ema:
                    #     policy = self.ema_model
                    policy.eval()

                    # run rollout (TASK SATISFACTION)
                    if ((self.epoch + 1) % cfg.training.rollout_every) == 0 and dist.get_rank() == 0:
                        dataset.empty_cache() # empty cache before running
                        val_dataset.empty_cache()
                        runner_log = env_runner.run(policy, cfg.policy, dataset.demos, mode="train")
                        runner_log.update(
                            env_runner.run(policy, cfg.policy, val_dataset.demos, mode="eval")
                        )
                        # log all
                        step_log.update(runner_log)

                    # run validation
                    if ((self.epoch + 1) % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    # batch = format_batch(batch)
                                    batch = dict_apply(batch, lambda x: x.to(device, dtype, non_blocking=True))
                                    if val_sampling_batch is None:
                                        val_sampling_batch = batch

                                    loss = self.model(
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
                                # log epoch average validation loss
                                if dist.get_rank() == 0:
                                    step_log['val_loss'] = val_loss


                    ## Run Experiment related Validation ## #TODO: as far as I see, this currently has no effect!
                    if ((self.epoch + 1) % cfg.training.model_evaluation_every) == 0 and dist.get_rank() == 0:
                        pred_act, pred_act_gr = self.model(
                            gt_trajectory=None,
                            rgb_obs=val_sampling_batch['obs'].get('rgb', None),
                            pcd_obs=val_sampling_batch['obs']['pcd'],
                            instruction=val_sampling_batch['obs'].get('instruction', None),
                            curr_gripper=val_sampling_batch['obs']['curr_gripper'],
                            run_inference=True,
                            feature_obs=val_sampling_batch['obs'].get('clip_features', None)
                        )
                        # log all
                        evaluation_log = criterion.compute_metrics(pred_act, pred_act_gr, val_sampling_batch, validation=True)
                        step_log.update(evaluation_log)

                    # sample on a training batch
                    if ((self.epoch + 1) % cfg.training.sample_every) == 0 and dist.get_rank() == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            train_sampling_batch = dict_apply(train_sampling_batch, lambda x: x.to(device, dtype, non_blocking=True))

                            pred_act, pred_act_gr = policy(
                                gt_trajectory=None,
                                rgb_obs=train_sampling_batch['obs'].get('rgb', None),
                                pcd_obs=train_sampling_batch['obs']['pcd'],
                                instruction=train_sampling_batch['obs'].get('instruction', None),
                                curr_gripper=train_sampling_batch['obs']['curr_gripper'],
                                run_inference=True,
                                feature_obs=train_sampling_batch['obs'].get('clip_features', None)
                            )
                            eval_log = criterion.compute_metrics(pred_act, pred_act_gr, train_sampling_batch)
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

        print ("training finished, now do the evaluation!")
        # add sleep here to ensure that all of the models are really saved
        time.sleep(10)
        if dist.get_rank() == 0:
            self.rollout(wandb_run=wandb_run)

    def rollout(self, wandb_run=None):
        cfg = copy.deepcopy(self.cfg)

        # get all checkpoints!
        filepath = self.output_dir + '/checkpoints'
        # now list all the checkpoints:
        checkpoint_list = os.listdir(filepath)

        # now go through all of them:
        all_checkpoints = []
        checkpoint_epoch = []
        for checkpoint in checkpoint_list:
            if (checkpoint[-5:]==".ckpt" and checkpoint[:6]=="epoch="):
                all_checkpoints.append(checkpoint)
                checkpoint_epoch.append(int(checkpoint.split('=')[-1].split('.')[0]))

        # now sort them:
        checkpoint_epoch = np.array(checkpoint_epoch)
        sorted_indices = np.argsort(checkpoint_epoch)
        all_checkpoints = np.array(all_checkpoints)[sorted_indices]
        checkpoint_epoch = checkpoint_epoch[sorted_indices]


        log_path = os.path.join(self.output_dir, 'eval_logs.json.txt')
        if wandb_run is None:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )

        with JsonLogger(log_path) as json_logger:

            for j in range(len(all_checkpoints)):
                if j>0 and checkpoint_epoch[j]==checkpoint_epoch[j-1]:
                    # skip if there are multiple checkpoints for the same epoch
                    continue

                # load the current checkpoint
                print ("Loading checkpoint: ", all_checkpoints[j])
                self.load_checkpoint(path=filepath + '/' + all_checkpoints[j])

                device = torch.device(cfg.training.device)
                self.model.to(device)
                policy = self.model
                policy.eval()

                env_runner = hydra.utils.instantiate(
                    cfg.env_runner,
                    output_dir=self.output_dir)
                dataset = dataset = self.get_dataset(cfg)
                val_dataset = dataset.get_test_dataset()
                self.model.set_mean_std(*dataset.get_mean_std(
                    relative_to_gripper=cfg.policy.relative,
                    quaternion_format=cfg.policy.quaternion_format)
                                        )

                with torch.no_grad():
                    env_runner.max_rrt_tries = 10
                    runner_log = env_runner.run(policy, cfg.policy, dataset.demos, mode="train")
                    runner_log.update(
                        env_runner.run(policy, cfg.policy, val_dataset.demos, mode="eval")
                    )
                    runner_log['epoch'] = int(checkpoint_epoch[j])
                    # log all
                    wandb_run.log(runner_log)
                    json_logger.log(runner_log)

        print ("Finished the evaluation!")

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

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    print("Device count", torch.cuda.device_count())
    local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    # DDP initialization
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    workspace = TrainingWorkspace(cfg)
    if cfg.mode == 'train':
        # we run the evaluation after the training - inside of the run loop
        workspace.run()
    elif cfg.mode == 'rollout':
        print("Rollout")
        workspace.rollout()
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")

if __name__ == "__main__":
    main()