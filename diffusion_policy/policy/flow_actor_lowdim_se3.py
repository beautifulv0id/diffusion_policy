import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion_policy.model.common.so3_util import log_map
from diffusion_policy.common.rotation_utils import normalise_quat
from diffusion_policy.model.common.se3_util import se3_from_rot_pos

from diffusion_policy.model.flow_matching.flow_matching_models import SE3LinearAttractorFlow

from diffusion_policy.model.common.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from diffusion_policy.model.obs_encoders.diffuser_actor_lowdim_encoder import DiffuserActorEncoder

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


class FlowActorSE3(BaseImagePolicy):

    def __init__(self,
                 embedding_dim=60,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 rotation_parametrization='so3',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 nkeypoints=10,
                 nhorizon=16,
                 t_switch=0.75,
                 relative=False,
                 ):
        super().__init__()
        assert rotation_parametrization == 'so3', "Only SO3 is supported"        
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.encoder = DiffuserActorEncoder(
            embedding_dim=embedding_dim,
            nhist=nhist,
            nkeypoints=nkeypoints,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
        )
        
        self.n_steps = diffusion_timesteps
        self.nhorizon = nhorizon
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)
        ## Flow Model ##
        self.t_switch = t_switch
        self.flow = SE3LinearAttractorFlow(t_switch=self.t_switch)
        self.generate_random_initial_pose = self.flow.generate_random_initial_pose
        self.flow_at_t = self.flow.flow_at_t
        self.vector_field_at_t = self.flow.vector_field_at_t
        self.step = self.flow.step

    def encode_inputs(self, keypoint_pcd,
                      curr_gripper):
        # Compute visual features/positional embeddings at different scales
        keypoint_feats = self.encoder.get_keypoint_embeddings()
        keypoint_feats = einops.repeat(keypoint_feats, 'npts c -> b npts c', b=keypoint_pcd.size(0))
        context_feats = keypoint_feats
        context = keypoint_pcd

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        return (
            context_feats, context,  # contextualized visual features
            adaln_gripper_feats,  # gripper history features
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, need_attn_weights=False):
        # Parse inputs
        (
            context_feats,
            context,
            adaln_gripper_feats,
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            adaln_gripper_feats=adaln_gripper_feats,
            need_attn_weights=need_attn_weights
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs, need_attn_weights=False):
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
                xt = torch.cat([pt, rt[...,:3,:2].reshape(-1, 1, 6)], dim=-1)
                out = self.policy_forward_pass(
                    xt,
                    time,
                    fixed_inputs
                )['pred']
                out = out[-1] # keep only last layer's output
                dp, dr = out[...,:3], out[...,3:6]   
                rt, pt = self.step(rt, pt, dr, dp, dt, time=s*dt)


        trajectory = se3_from_rot_pos(rt, pt)
        gripper_opennes = out[..., 6:7]

        return {
            'trajectory' : trajectory,
            'gripper_openess': gripper_opennes
        }

    def compute_trajectory(
        self,
        trajectory_mask,
        pcd_obs,
        curr_gripper,
        need_attn_weights=False
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = self.normalize_pos(pcd_obs)
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper, _ = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            pcd_obs, curr_gripper[:,:,:3,3]
        )

        # Condition on start-end pose
        B = curr_gripper.shape[0]
        D = 9
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=pcd_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        output = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs,
            need_attn_weights=need_attn_weights
        )

        trajectory = output['trajectory']            
        gripper_openess = output['gripper_openess']         

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory, res=gripper_openess)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        output['trajectory'] = trajectory

        return output

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal = signal.clone()
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == 'so3':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            H = se3_from_rot_pos(rot, signal[..., :3])
        return H, res

    def unconvert_rot(self, H, res=None):
        if self._rotation_parametrization == 'so3':
            quat = matrix_to_quaternion(H[..., :3, :3])
            pos = H[..., :3, 3]
            signal = torch.cat([pos, quat], dim=-1)
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
        pcd_obs,
        curr_gripper,
        run_inference=False,
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
                pcd_obs,
                curr_gripper,
                need_attn_weights=need_attn_weights
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = self.normalize_pos(pcd_obs)
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory, _ = self.convert_rot(gt_trajectory)
        curr_gripper, _ = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            pcd_obs, curr_gripper[:,:,:3,3]
        )

        p1 = gt_trajectory[:, :, :3, 3]
        r1 = gt_trajectory[:, :, :3, :3]

        # Add noise to the clean trajectories
        r0, p0 = self.generate_random_initial_pose(batch=gt_trajectory.shape[0], trj_steps=gt_trajectory.shape[1])
        r0, p0 = r0.to(gt_trajectory.device), p0.to(gt_trajectory.device)
        timesteps = torch.rand(gt_trajectory.shape[0], device=gt_trajectory.device)
        rt, pt = self.flow_at_t(r0, p0, r1, p1, timesteps)
        dr, dp = self.vector_field_at_t(r1,p1,rt,pt,timesteps)
        
        # Predict the noise residual
        trajectory_t = torch.cat([pt, rt[...,:3,:2].reshape(-1, 1, 6)], dim=-1)
        # Predict the noise residual
        pred = self.policy_forward_pass(
            trajectory_t, timesteps, fixed_inputs
        )['pred']

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:6]
            loss = (
                30 * F.mse_loss(trans, dp, reduction='mean')
                + 10 * F.mse_loss(rot, dr, reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 6:7]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], need_attn_weights=False) -> Dict[str, torch.Tensor]:
        trajectory_mask = torch.zeros(1, self.nhorizon, device=obs_dict['low_dim_pcd'].device)
        output = self.forward(
            gt_trajectory=None,
            trajectory_mask=trajectory_mask,
            pcd_obs=obs_dict['low_dim_pcd'],
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
            need_attn_weights=need_attn_weights
        )
        rlbench_action = output['trajectory'].clone()

        if rlbench_action.shape[-1] > 7:
            rlbench_action[..., 7] = rlbench_action[..., 7] > 0.5
            
        action = create_robomimic_from_rlbench_action(rlbench_action, quaternion_format = self._quaternion_format)
        result = {
            'rlbench_action' : rlbench_action,
            'action': action,
            'obs': obs_dict,
            'extra': {
                'act_gr_pred': output['trajectory'][..., 7],
            }
        }

        if need_attn_weights:
            result['attn_weights'] = output['attn_weights']
            result['attn_pcd'] = output['attn_pcd']
        return result
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            gt_trajectory=batch['action']['gt_trajectory'],
            trajectory_mask=None,
            pcd_obs=batch['obs']['low_dim_pcd'],
            curr_gripper=batch['obs']['curr_gripper'],
            run_inference=False,
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
        gt_act_gr = gt_trajectory[..., 7]

        out = self.predict_action(batch['obs'])
        action = out['action']
        pred_act_p = action['act_p']
        pred_act_r = action['act_r']
        pred_act_gr = action['act_gr']

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
        rotation_dim = 6  # continuous 6D
        rotation_dim_out = 3

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
            nn.Linear(embedding_dim, rotation_dim_out)
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
                context_feats, context, adaln_gripper_feats,
                need_attn_weights=False):
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
            need_attn_weights=need_attn_weights
        )

        pos_pred, rot_pred, openess_pred = output['pred']
        output['pred'] = [torch.cat((pos_pred, rot_pred, openess_pred), -1)]
        return output
    
    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        need_attn_weights=False):
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
            diff_ts=time_embs,
            need_weights=need_attn_weights
        )

        if need_attn_weights:
            gripper_features, attn_weights = gripper_features

        gripper_features = gripper_features[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper
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

    def predict_pos(self, features, rel_pos, time_embs, num_gripper):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
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