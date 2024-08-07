from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from diffusion_policy.model.common.so3_util import log_map

from diffusion_policy.common.rotation_utils import (
    get_ortho6d_from_rotation_matrix,
    compute_rotation_matrix_from_ortho6d,
    normalise_quat,
    )

from diffusion_policy.model.obs_encoders.diffusion_policy_image_encoder import MultiImageObsEncoder

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            obs_encoder : MultiImageObsEncoder,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim,
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            rotation_parametrization='6D',
            quaternion_format='xyzw',
            gripper_loc_bounds=None,
            normalize=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.model = model
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.normalize = normalize
    

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # assert 'obs' in obs_dict
        # assert 'past_action' not in obs_dict # not implemented yet
        # nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        nobs = obs_dict
        curr_gripper = nobs['curr_gripper'][...,:8]
        rgb = nobs['rgb']

        # normalize position
        if self.normalize:
            curr_gripper[...,:3] = self.normalize_pos(curr_gripper[...,:3])

        # convert rotation
        curr_gripper = self.convert_rot(curr_gripper)

        # encode observation
        with torch.no_grad():
            obs_feat = self.obs_encoder(curr_gripper, rgb)
        B, _, Do = obs_feat.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        # condition throught global feature
        global_cond = obs_feat.reshape(obs_feat.shape[0], -1)
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = naction_pred

        # action_pred = self.normalizer['action'].unnormalize(naction_pred)
        if self._rotation_parametrization == 'quat':
            action_pred[..., 3:7] = normalise_quat(action_pred[..., 3:7])

        action_pred = self.unconvert_rot(action_pred)

        if self.normalize:
            action_pred[...,:3] = self.unnormalize_pos(action_pred[...,:3])

        if action_pred.shape[-1] > 7:
            action_pred[..., 7] = action_pred[..., 7].sigmoid()

        rlbench_action = action_pred.clone()
        if rlbench_action.shape[-1] > 7:
            rlbench_action[..., 7] = rlbench_action[..., 7] > 0.5
        
        result = {
            'rlbench_action': rlbench_action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # nbatch = self.normalizer.normalize(batch)
        nbatch = batch
        obs = nbatch['obs']
        action = nbatch['action']['gt_trajectory']
        action = action[:,:self.horizon, :8]

        curr_gripper = obs['curr_gripper'][...,:8]
        rgb = obs['rgb']

        # normalize position
        if self.normalize:
            action[...,:3] = self.normalize_pos(action[...,:3])
            curr_gripper[...,:3] = self.normalize_pos(curr_gripper[...,:3])

        # convert rotation
        action = self.convert_rot(action)
        curr_gripper = self.convert_rot(curr_gripper)

        # encode observation
        with torch.no_grad():
            obs_feat = self.obs_encoder(curr_gripper, rgb)

        # handle different ways of passing observation
        local_cond = None
        trajectory = action
        global_cond = obs_feat.reshape(obs_feat.shape[0], -1)

        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    @torch.no_grad()
    def evaluate(self, batch, validation=False):
        log_dict = {}

        gt_action = batch['action']
        gt_trajectory = gt_action['gt_trajectory']
        gt_act_p = gt_trajectory[..., :3]
        gt_act_r = gt_trajectory[..., 3:7]
        if self._quaternion_format == 'xyzw':
            gt_act_r = gt_act_r[..., (3, 0, 1, 2)]
        gt_act_r = normalise_quat(gt_act_r)
        gt_act_r = quaternion_to_matrix(gt_act_r)
        gt_act_gr = gt_trajectory[..., 7]

        out = self.predict_action(batch['obs'])
        action = out['rlbench_action']
        pos, rot, gr = action.split([3, 4, 1], dim=-1)
        pred_act_p = pos
        pred_act_r = quaternion_to_matrix(rot)
        pred_act_gr = gr.squeeze(-1)

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