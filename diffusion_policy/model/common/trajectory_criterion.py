import torch
from diffusion_policy.common.rotation_utils import normalise_quat
from pytorch3d.transforms import quaternion_to_matrix
from diffusion_policy.common.so3_util import log_map

class TrajectoryCriterion:
    def __init__(self, quaternion_format='xyzw'):
        self._quaternion_format = quaternion_format

    def compute_loss(self, pred, gt=None):
        return pred

    def compute_metrics(self, pred_act, batch, validation=False):
        log_dict = {}

        tasks= batch['obs']['task']
        gt_trajectory = batch['action']['gt_trajectory']
        gt_act_p = gt_trajectory[..., :3]
        gt_act_r = gt_trajectory[..., 3:7]
        gt_act_gr = gt_trajectory[..., 7:8]
        if self._quaternion_format == 'xyzw':
            gt_act_r = gt_act_r[..., (3, 0, 1, 2)]
        gt_act_r = normalise_quat(gt_act_r)
        gt_act_r = quaternion_to_matrix(gt_act_r)

        pred_act_p = pred_act[..., :3]
        pred_act_r = pred_act[..., 3:7]
        pred_act_gr = pred_act[..., 7:8]
        if self._quaternion_format == 'xyzw':
            pred_act_r = pred_act_r[..., (3, 0, 1, 2)]
        pred_act_r = normalise_quat(pred_act_r)
        pred_act_r = quaternion_to_matrix(pred_act_r)

        pos_error = torch.nn.functional.mse_loss(pred_act_p, gt_act_p, reduction='none').mean(dim=-1)

        R_inv_gt = torch.transpose(gt_act_r, -1, -2)
        relative_R = torch.matmul(R_inv_gt, pred_act_r)
        angle_error = log_map(relative_R)
        rot_error = torch.nn.functional.mse_loss(angle_error, torch.zeros_like(angle_error), reduction='none').mean(dim=-1)
        gr_error = torch.nn.functional.l1_loss(pred_act_gr, gt_act_gr, reduction='none').mean(dim=-1)

        prefix = 'val_' if validation else 'train_'

        unique_tasks = set(tasks)
        for task in unique_tasks:
            task_mask = [t == task for t in tasks]            
            task_pos_error = pos_error[task_mask].mean()  # Average over samples of this task
            task_rot_error = rot_error[task_mask].mean()  # Average over samples of this task
            task_gr_error = gr_error[task_mask].mean()  # Average over samples of this task

            # Add task-specific metrics to log_dict
            log_dict[f'{task}/{prefix}gripper_l1_loss'] = task_gr_error
            log_dict[f'{task}/{prefix}position_mse_error'] = task_pos_error
            log_dict[f'{task}/{prefix}rotation_mse_error'] = task_rot_error


        return log_dict