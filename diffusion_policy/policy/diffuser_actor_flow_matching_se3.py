import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion_policy.model.common.so3_util import log_map, normal_so3
from diffusion_policy.model.flow_matching.flow_matching_models import SE3LinearAttractorFlow

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
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from diffusion_policy.model.common.so3_util import log_map
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import Dict


class DiffuserActorFLowMatching(BaseImagePolicy):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 rotation_parametrization='so3',
                 quaternion_format='xyzw',
                 num_inference_steps=100,
                 nhist=3,
                 nhorizon=16,
                 relative=False,
                 lang_enhanced=False,
                 t_switch=0.75):
        super().__init__()
        assert rotation_parametrization == 'so3', "Only SO3 is supported"
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
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

        self.n_steps = num_inference_steps
        self.nhorizon = nhorizon
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

        ## Flow Model ##
        self.t_switch = t_switch
        self.flow = SE3LinearAttractorFlow()
        self.generate_random_initial_pose = self.flow.generate_random_initial_pose
        self.flow_at_t = self.flow.flow_at_t
        self.vector_field_at_t = self.flow.vector_field_at_t
        self.step = self.flow.step

    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

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
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos  # sampled visual features
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        # Random trajectory, conditioned on start-end
        B, L, D = condition_data.size()
        device = condition_data.device
        # Iterative denoising
        with torch.no_grad():
            dt = 1.0 / self.n_steps
            r0, p0 = self.generate_random_initial_pose(batch=B, trj_steps=L)
            r0, p0 = r0.to(device), p0.to(device)
            rt, pt = r0, p0
            for s in range(0, self.n_steps):
                time = s*dt*torch.ones_like(pt[:, 0, 0], device=device)
                xt = torch.cat([pt, rt.flatten(-2)], -1)
                out = self.policy_forward_pass(
                    xt,
                    time,
                    fixed_inputs
                )
                out = out[-1] # keep only last layer's output
                dr, dp = out[...,:3], out[...,3:6]   
                rt, pt = self.step(rt, pt, dr, dp, dt, time=s*dt)


        trajectory = torch.cat([pt, rt.flatten(-2)], -1)
        
        if self._rotation_parametrization == 'so3':
            trajectory = torch.cat([trajectory, out[..., 6:]], -1)
        if self._rotation_parametrization == 'quat':
            trajectory = torch.cat([trajectory, out[..., 7:]], -1)
        if self._rotation_parametrization == '6D':
            trajectory = torch.cat([trajectory, out[..., 9:]], -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization == 'quat':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

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
        if self._rotation_parametrization != 'quat':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot_mat = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if self._rotation_parametrization == '6D':
                if len(rot_mat.shape) == 4:
                    B, L, D1, D2 = rot_mat.shape
                    rot_mat = rot_mat.reshape(B * L, D1, D2)
                    rot_6d = get_ortho6d_from_rotation_matrix(rot_mat)
                    rot_6d = rot_6d.reshape(B, L, 6)
                else:
                    rot_6d = get_ortho6d_from_rotation_matrix(rot_mat)
                signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if self._rotation_parametrization == 'so3':
                signal = torch.cat([signal[..., :3], rot_mat.flatten(-2)], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization != 'quat':
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
            if self._rotation_parametrization == 'so3':
                res = signal[..., 12:] if signal.size(-1) > 12 else None
                if len(signal.shape) == 3:
                    B, L, _ = signal.shape
                    rot = signal[..., 3:12].reshape(B, L, 3, 3)
                    quat = matrix_to_quaternion(rot)
                else:
                    rot = signal[..., 3:12].reshape(-1, 3, 3)
                    quat = matrix_to_quaternion(rot)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        
        return signal

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False
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
        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:8]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        B, L, _ = gt_trajectory.shape
        p1 = gt_trajectory[:, :, :3]
        r1 = gt_trajectory[:, :, 3:12].reshape(B, L, 3, 3)

        # Add noise to the clean trajectories
        r0, p0 = self.generate_random_initial_pose(batch=gt_trajectory.shape[0], trj_steps=gt_trajectory.shape[1])
        r0, p0 = r0.to(gt_trajectory.device), p0.to(gt_trajectory.device)
        time = torch.rand(gt_trajectory.shape[0], device=gt_trajectory.device)
        rt, pt = self.flow_at_t(r0, p0, r1, p1, time)
        dr, dp = self.vector_field_at_t(r1,p1,rt,pt,time)

        # Predict the noise residual
        trajectory_t = torch.cat([pt, rt.flatten(-2)], -1)
        pred = self.policy_forward_pass(
            trajectory_t, time, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            rot = layer_pred[..., :3]
            trans = layer_pred[..., 3:6]
            loss = (
                30 * F.mse_loss(trans, dp, reduction='mean') + 
                + 10 * F.mse_loss(rot, dr, reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 6:7]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trajectory_mask = torch.zeros(1, self.nhorizon, device=obs_dict['rgb'].device)
        pred = self.forward(
            gt_trajectory=None,
            trajectory_mask=trajectory_mask,
            rgb_obs=obs_dict['rgb'],
            pcd_obs=obs_dict['pcd'],
            instruction=None,
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
        )

        action = create_robomimic_from_rlbench_action(pred, quaternion_format = 'wxyz')
        result = {
            'action': action,
            'obs': obs_dict,
            'extra': {
                'act_gr_pred': pred[..., 7],
                # 'act_ic_pred': pred[..., 8]
            }
        }
        return result
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            gt_trajectory=batch['action']['gt_trajectory'],
            trajectory_mask=None,
            rgb_obs=batch['obs']['rgb'],
            pcd_obs=batch['obs']['pcd'],
            instruction=None,
            curr_gripper=batch['obs']['curr_gripper'],
            run_inference=False,
        )


    def evaluate(self, batch):
        log_dict = {}

        gt_action = batch['action']
        gt_act_p = gt_action['act_p']
        gt_act_r = gt_action['act_r']

        out = self.predict_action(batch['obs'])
        action = out['action']
        pred_act_p = action['act_p']
        pred_act_r = action['act_r']
        pred_act_gr = out['extra']['act_gr_pred']
        # pred_act_ic = out['extra']['act_ic_pred']

        pos_error = torch.nn.functional.mse_loss(pred_act_p, gt_act_p)

        R_inv_gt = torch.transpose(gt_act_r, -1, -2)
        relative_R = torch.matmul(R_inv_gt, pred_act_r)
        angle_error = log_map(relative_R)
        rot_error = torch.nn.functional.mse_loss(angle_error, torch.zeros_like(angle_error))

        gt_act_gr = gt_action['act_gr']
        gr_error = torch.nn.functional.mse_loss(pred_act_gr, gt_act_gr)
        log_dict['train_gripper_mse_error'] = gr_error.item()

        # gt_act_ic = gt_action['act_ic']
        # ic_error = torch.nn.functional.mse_loss(pred_act_ic, gt_act_ic)
        # log_dict['train_ignore_collisions_mse_error'] = ic_error.item()

        log_dict['train_position_mse_error'] = pos_error.item()
        log_dict['train_rotation_mse_error'] = rot_error.item()

        return log_dict
class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='so3',
                 nhist=3,
                 lang_enhanced=False):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        if '6D' in rotation_parametrization:
            rotation_out_dim = 6  # continuous 6D
            rotation_in_dim = 6
        elif 'quat' in rotation_parametrization:
            rotation_out_dim = 4  # quaternion
            rotation_in_dim = 6
        elif 'so3' in rotation_parametrization:
            rotation_out_dim = 3
            rotation_in_dim = 9
        else:
            raise ValueError(f"Unknown rotation parametrization: {rotation_parametrization}")

        # Encoders
        self.traj_encoder = nn.Linear(3+rotation_in_dim, embedding_dim)
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
            nn.Linear(embedding_dim, rotation_out_dim)
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
                fps_feats, fps_pos):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
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
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats):
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

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

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

        return position, rotation, openess

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 16
    nhist = 3
    image_size = (256, 256)

    model = DiffuserActorFLowMatching(
        backbone="clip",
        image_size=(256, 256),
        embedding_dim=192,
        num_vis_ins_attn_layers=2,
        use_instruction=False,
        fps_subsampling_factor=5,
        gripper_loc_bounds=[[-1, -1, -1], [1, 1, 1]],
        rotation_parametrization='so3',
        quaternion_format='xyzw',
        num_inference_steps=100,
        nhist=nhist,
        nhorizon=horizon,
        relative=False,
        lang_enhanced=False
    )

    model.to(device)


    batch = {
        'action': {
            'gt_trajectory': torch.randn(1, horizon, 3+4+2),
            'act_p': torch.randn(1, horizon, 3),
            'act_r': torch.randn(1, horizon, 3, 3),
            'act_gr': torch.randn(1, horizon),
            'act_ic': torch.randn(1, horizon)
        },
        'obs': {
            'rgb': torch.randn(1, 2, 3, 256, 256),
            'pcd': torch.randn(1, 2, 3, 256, 256),
            'curr_gripper': torch.randn(1, nhist, 3+4+2)
        }
    }

    batch = dict_apply(batch, lambda x: x.to(device))

    model.compute_loss(batch)
    print("Loss computed successfully")
    model.predict_action(batch['obs'])
    print("Action predicted successfully")

    out = model.evaluate(batch)
    print("Evaluation done successfully")
    print(out)
    print("DiffuserActor test passed")

if __name__ == "__main__":
    test()
    print("Test passed")