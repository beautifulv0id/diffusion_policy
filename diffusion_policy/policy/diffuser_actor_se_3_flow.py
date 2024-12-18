import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from diffusion_policy.model.obs_encoders.diffuser_actor_encoder import DiffuserActorEncoder

from diffusion_policy.model.common.layers import ParallelAttention
from diffusion_policy.model.common.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)

from diffusion_policy.common.rotation_utils import (
    get_ortho6d_from_rotation_matrix,
    compute_rotation_matrix_from_ortho6d,
    normalise_quat
    )

from diffusion_policy.common.rlbench_util import create_robomimic_from_rlbench_action
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, quaternion_invert, quaternion_multiply, quaternion_apply
from diffusion_policy.common.so3_util import log_map, se3_inverse
from diffusion_policy.common.se3_util import se3_from_rot_pos
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import Dict

from diffusion_policy.model.common.workspace_cropping import crop_to_workspace
from diffusion_policy.model.obs_encoders.feature_pcd_encoder import FeaturePCDEncoder

from diffusion_policy.common.rlbench_util import convert_rlbench_action, unconvert_rlbench_action

from torch import einsum

from geo3dattn.policy.se3_flowmatching.common.se3_flowmatching import RectifiedLinearFlow
import math

def pos_quat_apply(pq1, pq2):
    p1, q1 = pq1[..., :3], pq1[..., 3:7]
    p2, q2 = pq2[..., :3], pq2[..., 3:7]
    p = p1 + quaternion_apply(q1, p2)
    q = quaternion_multiply(q1, q2)
    pq = torch.cat((p, q), -1)
    if pq2.size(-1) > 7:
        ret = pq2[..., 7:]
        pq = torch.cat((pq, ret), -1)
    return pq
    

class DiffuserActor(BaseImagePolicy):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 feature_res="res2",
                 workspace_bounds=None,
                 diffusion_timesteps=100,
                 scaling_factor=1.0,
                 nhist=3,
                 nhorizon=16,
                 relative=False,
                 lang_enhanced=False,
                 use_mask=False,):
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
        self.feature_pcd_encoder = FeaturePCDEncoder(
            backbone=backbone,
            feature_res=feature_res
        )
        self.feature_pcd_up = nn.Linear(self.feature_pcd_encoder.out_dim, embedding_dim)
        self.encoder = DiffuserActorEncoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )

        #Flow Model
        # TODO - mean and std has to be set!
        self.flow = RectifiedLinearFlow(n_action_steps=1, num_steps=diffusion_timesteps)
        # TODO should be improved - empirical guess here:
        std = torch.Tensor([0.6, 0.6, 0.6, math.pi, math.pi, math.pi])[None, ...].repeat(nhorizon, 1)
        mean = torch.Tensor([0, 0, 0, 0, 0, 0])[None, ...].repeat(nhorizon, 1)
        self.flow.set_mean_std(mean, std)

        self.scaling_factor = torch.tensor(scaling_factor)
        self.n_steps = diffusion_timesteps
        self.nhorizon = nhorizon
        self.use_mask = use_mask
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds) if gripper_loc_bounds is not None else None
        if workspace_bounds is not None:
            self.register_buffer("workspace_bounds", torch.tensor(workspace_bounds, requires_grad=False))
        else:
            self.workspace_bounds = None

    def encode_inputs(self, context_feats, context, instruction,
                      curr_gripper, mask=None):
        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pcd = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            context
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pcd  # sampled visual features
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, need_attn_weights=False):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pcd
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pcd=fps_pcd,
            need_attn_weights=need_attn_weights
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs, need_attn_weights=False):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        # TODO - do not understand here why they add noise, I think in training it is zeros, but maybe does not matter anyways,...
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        # Iterative denoising WITH SE3 FLOW
        with torch.no_grad():
            # SE3 FLOW
            B = condition_data.shape[0]
            at = self.flow.generate_random_initial_pose(B)
            start_3ddac = self.flow_convention_to_3ddact_convention(at.clone())
            trajectory_flow = torch.where(
                condition_mask, start_3ddac, noise
            )
            for s in range(0, self.flow.num_steps):
                step = s * torch.ones_like(at[:, 0, 0])
                polic_output = self.policy_forward_pass(
                    trajectory_flow,
                    s * torch.ones(len(trajectory)).to(trajectory_flow.device).long(),
                    fixed_inputs,
                    need_attn_weights=need_attn_weights
                )
                model_out = polic_output['pred'][-1]  # keep only last layer's output
                d_act = self.three_ddact_convention_to_flow(model_out[..., :9])
                at = self.flow.step(at, d_act, s)
                trajectory_flow = self.flow_convention_to_3ddact_convention(at.clone())

        trajectory = torch.cat((trajectory_flow, model_out[..., 9:]), -1)

        # # Iterative denoising
        # # TODO - also add torch no grad here???
        # timesteps = self.position_noise_scheduler.timesteps
        # for t in timesteps:
        #     polic_output = self.policy_forward_pass(
        #         trajectory,
        #         t * torch.ones(len(trajectory)).to(trajectory.device).long(),
        #         fixed_inputs,
        #         need_attn_weights=need_attn_weights
        #     )
        #     model_out = polic_output['pred'][-1]# keep only last layer's output
        #     pos = self.position_noise_scheduler.step(
        #         model_out[..., :3], t, trajectory[..., :3]
        #     ).prev_sample
        #     rot = self.rotation_noise_scheduler.step(
        #         model_out[..., 3:9], t, trajectory[..., 3:9]
        #     ).prev_sample
        #     trajectory = torch.cat((pos, rot), -1)
        #
        # # TODO- interesting, seems gripper state only taken from last reading,..
        # trajectory = torch.cat((trajectory, model_out[..., 9:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        feature_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        mask_obs=None,
        need_attn_weights=False
    ):       
        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            feature_obs, pcd_obs, instruction, curr_gripper, mask_obs
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=feature_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        return self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs,
            need_attn_weights=need_attn_weights
        )

    def normalize_pos(self, x):
        x = x.clone()
        if self.gripper_loc_bounds is None:
            return x * self.scaling_factor
        pos_min = self.gripper_loc_bounds[0].float().to(x.device)
        pos_max = self.gripper_loc_bounds[1].float().to(x.device)
        x[...,:3] = (x[...,:3] - pos_min) / (pos_max - pos_min) * 2.0 - 1.0
        return x

    def unnormalize_pos(self, x):
        x = x.clone()
        if self.gripper_loc_bounds is None:
            return x / self.scaling_factor
        pos_min = self.gripper_loc_bounds[0].float().to(x.device)
        pos_max = self.gripper_loc_bounds[1].float().to(x.device)
        x[...,:3] = (x[...,:3] + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
        return x

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

    def convert_rot_flow(self, signal):
        signal = signal.clone()
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        # The following code expects wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
        rot = quaternion_to_matrix(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        H = se3_from_rot_pos(rot, signal[..., :3])
        return H, res

    def unconvert_rot_flow(self, H, res=None):
        quat = matrix_to_quaternion(H[..., :3, :3])
        pos = H[..., :3, 3]
        signal = torch.cat([pos, quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        # The above code handled wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    def unconvert_rot_flow_rotation_only(self, H):
        quat = matrix_to_quaternion(H[..., :3, :3])
        signal = quat
        # The above code handled wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 0:4] = signal[..., (1, 2, 3, 0)]
        return signal

    def flow_convention_to_3ddact_convention(self, vec):
        # idea here: vector format to tranlation and rotation matrix to quaternion to 6D representation
        flow_to_pose = self.flow._vector_to_pose(vec)
        return self.convert_rot(
        torch.cat((flow_to_pose[0], self.unconvert_rot_flow_rotation_only(flow_to_pose[1])), dim=-1))

    def three_ddact_convention_to_flow(self, three_ddact_vec):
        # idea here: vector format to tranlation and rotation matrix to quaternion to 6D rot matrix and then to vector
        pos_quat_repr = self.unconvert_rot(three_ddact_vec)
        homogeneous = self.convert_rot_flow(pos_quat_repr)[0]
        return self.flow._pose_to_vector(homogeneous[..., :3, -1], homogeneous[..., :3, :3])


    def convert2gripper(self, x):
        x = x.clone()
        x[...,:3] = x[...,:3] - self.rel_to
        return x
        
    def convert2world(self, x):
        x = x.clone()
        x[...,:3] = x[...,:3] + self.rel_to
        return x
    
    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        feature_obs=None,
        run_inference=False,
        mask_obs=None,
        need_attn_weights=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.
        """
        if feature_obs is None:
            feature_obs, pcd_obs = self.feature_pcd_encoder(rgb_obs, pcd_obs)
            feature_obs = self.feature_pcd_up(feature_obs)
            if self.workspace_bounds is not None:
                pcd_obs, feature_obs = crop_to_workspace(pcd_obs, feature_obs, self.workspace_bounds, self.max_pcd_points)       

        # Normalize all pos
        if gt_trajectory is not None:
            gt_trajectory = self.normalize_pos(gt_trajectory)
        pcd_obs = self.normalize_pos(pcd_obs)
        curr_gripper = self.normalize_pos(curr_gripper)
        curr_gripper = curr_gripper[..., :7]

        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:8]
            gt_trajectory = gt_trajectory[..., :7]

        if self._relative:
            self.rel_to = curr_gripper[:, -1:, :3]
            pcd_obs = self.convert2gripper(pcd_obs)
            curr_gripper = self.convert2gripper(curr_gripper)
            if gt_trajectory is not None:
                gt_trajectory = self.convert2gripper(gt_trajectory)

        # Convert rotation parametrization
        curr_gripper = self.convert_rot(curr_gripper)
        if gt_trajectory is not None:
            gt_trajectory_flow = self.convert_rot_flow(gt_trajectory.clone())
            gt_trajectory = self.convert_rot(gt_trajectory)

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                feature_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                mask_obs,
                need_attn_weights=need_attn_weights
            )

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            feature_obs, pcd_obs, instruction, curr_gripper, mask_obs
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # PREPARE FLOW SPECIFICS!
        batch_size = pcd_obs.shape[0]
        device, dtype = pcd_obs.device, pcd_obs.dtype
        act_vector = self.flow._pose_to_vector(gt_trajectory_flow[0][...,:3, -1], gt_trajectory_flow[0][...,:3, :3])

        # 2. Compute Flow Matching Variables
        a1 = act_vector
        a0 = self.flow.generate_random_initial_pose(batch_size)
        time = torch.randint(0, self.flow.num_steps, (batch_size,)).to(device=gt_trajectory.device, dtype=dtype).long()

        at = self.flow.flow_at_t(a0, a1, time)
        target = self.flow.vector_field_at_t(a0, a1, at, time)

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()
        # timesteps = time

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )

        # convert the flow info into the right format - what is done here
        # 1) from axis angle to rotation matrix, then rotation matrix to quaternion, then quternion to 6D
        noisy_traj_flow_format = self.flow_convention_to_3ddact_convention(at)


        noisy_trajectory = torch.cat((pos, rot), -1)
        # TODO - need to understand this step,... why is this done?
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        noisy_traj_flow_format[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # # Predict the noise residual - originally with 3DDACTOR sampling
        # pred = self.policy_forward_pass(
        #     noisy_trajectory, timesteps, fixed_inputs
        # )['pred']

        # NOW WITH FLOW!
        pred = self.policy_forward_pass(
            noisy_traj_flow_format, time, fixed_inputs
        )['pred']


        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            # # this was original loss 3DDACTOR
            # loss = (
            #     30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
            #     + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
            # )
            targets_fm = self.flow_convention_to_3ddact_convention(target)
            loss = (
                30 * F.l1_loss(trans, targets_fm[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, targets_fm[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], need_attn_weights=False) -> Dict[str, torch.Tensor]:
        trajectory_mask = torch.zeros(1, self.nhorizon, device=obs_dict['pcd'].device)
        trajectory = self.forward(
            gt_trajectory=None,
            trajectory_mask=trajectory_mask,
            rgb_obs=obs_dict.get('rgb', None),
            pcd_obs=obs_dict['pcd'],
            instruction=None,
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
            mask_obs=obs_dict.get('mask', None),
            feature_obs=None,
            need_attn_weights=need_attn_weights,
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)

        if self._relative:
            trajectory = self.convert2world(trajectory)

        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        output = dict()
        output['trajectory'] = trajectory

        return output
    
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            gt_trajectory=batch['action']['gt_trajectory'],
            trajectory_mask=None,
            rgb_obs=batch['obs'].get('rgb', None),
            pcd_obs=batch['obs']['pcd'],
            instruction=None,
            curr_gripper=batch['obs']['curr_gripper'],
            run_inference=False,
            mask_obs=batch['obs'].get('mask', None)
        )


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
        gt_act_gr = gt_trajectory[..., 7:8]

        out = self.predict_action(batch['obs'])
        trajectory = out['trajectory']
        pred_act_r, pred_act_p, pred_act_gr = convert_rlbench_action(trajectory)

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
    
    
class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3,
                 lang_enhanced=False):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # Encoders
        self.traj_encoder = nn.Linear(9, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pcd, need_attn_weights=False):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pcd: (B, N, 3)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        output = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pcd,
            instr_feats,
            need_attn_weights=need_attn_weights
        )

        pos_pred, rot_pred, openess_pred = output['pred']
        output['pred'] = [torch.cat((pos_pred, rot_pred, openess_pred), -1)]
        return output
    
    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_context_pcd,
                        instr_feats, need_attn_weights=False):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)
        sampled_rel_context_pos = self.relative_pe_layer(sampled_context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs,
            need_weights=need_attn_weights
        )

        if need_attn_weights:
            gripper_features, attn_weights = gripper_features

        gripper_features = gripper_features[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        output = {
            'pred': [position, rotation, openess],
        }

        if need_attn_weights:
            output['attn_weights'] = attn_weights
            output['attn_pcd'] = context_pcd

        return output

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

def test():
    from diffusion_policy.common.pytorch_util import dict_apply
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, random_quaternions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 1
    nhist = 3
    image_size = (256, 256)

    model = DiffuserActor(
        backbone="clip",
        image_size=(256, 256),
        embedding_dim=192,
        num_vis_ins_attn_layers=2,
        use_instruction=False,
        fps_subsampling_factor=5,
        gripper_loc_bounds=[[-1, -1, -1], [1, 1, 1]],
        rotation_parametrization='6D',
        quaternion_format='xyzw',
        diffusion_timesteps=100,
        nhist=nhist,
        nhorizon=horizon,
        relative=True,
        lang_enhanced=False
    )

    model.to(device)

    gt_trajectory = torch.cat((torch.randn(1, horizon, 3), random_quaternions(horizon).reshape(1, horizon, 4), torch.randn(1, horizon, 2)), -1)
    curr_gripper = torch.cat((torch.randn(1, nhist, 3), random_quaternions(nhist).reshape(1, nhist, 4), torch.randn(1, nhist, 2)), -1)


    batch = {
        'action': {
            'gt_trajectory': gt_trajectory,
            'act_p': torch.randn(1, horizon, 3),
            'act_r': torch.randn(1, horizon, 3, 3),
            'act_gr': torch.randn(1, horizon),
            'act_ic': torch.randn(1, horizon)
        },
        'obs': {
            'rgb': torch.randn(1, 2, 3, 256, 256),
            'pcd': torch.randn(1, 2, 3, 256, 256),
            'curr_gripper': curr_gripper
        }
    }

    batch = dict_apply(batch, lambda x: x.to(device))

    rot = random_quaternions(1, device=device).reshape(1, 1, 4)
    pos = torch.randn(1, 3, device=device).reshape(1, 1, 3)
    pq = torch.cat((pos, rot), -1)
    rot_inv = quaternion_invert(rot)
    pos_inv = -quaternion_apply(rot_inv, pos)
    pq_inv = torch.cat((pos_inv, rot_inv), -1)

    def rotate_batch(batch, pq):
        this_batch = dict_apply(batch, lambda x: x.clone())
        pcd = this_batch['obs']['pcd'].clone()
        curr_gripper = this_batch['obs']['curr_gripper'].clone()
        trajectory = this_batch['action']['gt_trajectory'].clone()
        inv_rot = pq[:,:, 3:7]
        inv_pos = pq[:,:, :3]

        b, v, c, h, w = pcd.shape
        pcd = einops.rearrange(pcd, 'b v c h w->b (v h w) c')
        pcd = quaternion_apply(inv_rot, pcd) + inv_pos
        pcd = einops.rearrange(pcd, 'b (v h w) c->b v c h w', v=v, h=h, w=w)
        curr_gripper = pos_quat_apply(pq, curr_gripper)
        if trajectory is not None:
            trajectory = pos_quat_apply(pq, trajectory)
            this_batch['action']['gt_trajectory'] = trajectory
        this_batch['obs']['pcd'] = pcd
        this_batch['obs']['curr_gripper'] = curr_gripper  
        return this_batch

    rotated_batch = rotate_batch(batch, pq)
    back_rotated_batch = rotate_batch(rotated_batch, pq_inv)

    # compare batch and back rotated batch
    print('curr_gripper', torch.abs(batch['obs']['curr_gripper']-back_rotated_batch['obs']['curr_gripper']).mean())
    print('pcd', torch.abs(batch['obs']['pcd']-back_rotated_batch['obs']['pcd']).mean())
    print('gt_trajectory', torch.abs(batch['action']['gt_trajectory']-back_rotated_batch['action']['gt_trajectory']).mean())


    torch.random.manual_seed(0)
    loss = model.compute_loss(batch)
    print("Loss computed successfully: ", loss.item())
    torch.random.manual_seed(0)
    action = model.predict_action(batch['obs'])['rlbench_action']
    print("Action predicted successfully", action.detach().cpu().numpy())

    print("Rotation results:")
    torch.random.manual_seed(0)
    loss = model.compute_loss(rotated_batch)
    print("Loss computed successfully: ", loss.item())
    torch.random.manual_seed(0)
    action = model.predict_action(rotated_batch['obs'])['rlbench_action']
    print("Action predicted successfully", action.detach().cpu().numpy())

    # out = model.evaluate(batch)
    # print("Evaluation done successfully")
    # print(out)
    # print("DiffuserActor test passed")

if __name__ == "__main__":
    test()
    print("Test passed")