import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion_policy.model.common.so3_util import log_map, se3_inverse
from diffusion_policy.model.common.se3_util import se3_from_rot_pos
from diffusion_policy.common.rotation_utils import normalise_quat
from diffusion_policy.model.flow_matching.flow_matching_models import SE3LinearAttractorFlow, RectifiedLinearFlow


from diffusion_policy.model.invariant_tranformers.invariant_point_transformer import InvariantPointTransformer
from diffusion_policy.model.invariant_tranformers.geometry_invariant_attention import InvariantPointAttention

from diffusion_policy.model.obs_encoders.diffuser_actor_pose_invariant_encoder_v2 import DiffuserActorEncoder

from diffusion_policy.model.common.layers import ParallelAttention
from diffusion_policy.model.common.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)
from torch import einsum


from diffusion_policy.common.rlbench_util import create_robomimic_from_rlbench_action
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from diffusion_policy.model.common.so3_util import log_map
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import Dict
from diffusion_policy.model.common.geometry_invariant_transformer import GeometryInvariantTransformer


class DiffuserActor(BaseImagePolicy):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 fps_subsampling_factor=5,
                 scaling_factor=3.0,
                 rotation_parametrization='so3',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 nhorizon=16,
                 t_switch=0.75,
                 use_mask=False,
                 relative=False,
                 causal_attn=True,
                 gripper_loc_bounds=None
                 ):
        super().__init__()
        assert rotation_parametrization == 'so3', "Only SO3 is supported"        
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self.encoder = DiffuserActorEncoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            causal_attn=causal_attn
        )
        self.n_steps = diffusion_timesteps
        self.nhorizon = nhorizon
        self.scaling_factor = torch.tensor(scaling_factor)
        ## Flow Model ##
        self.t_switch = t_switch
        self.flow = SE3LinearAttractorFlow(t_switch=self.t_switch)
        self.generate_random_initial_pose = self.flow.generate_random_initial_pose
        self.flow_at_t = self.flow.flow_at_t
        self.vector_field_at_t = self.flow.vector_field_at_t
        self.step = self.flow.step
        self._relative = relative
        self.use_mask = use_mask
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds) if gripper_loc_bounds is not None else None

    def encode_inputs(self, visible_rgb, visible_pcd,
                      curr_gripper, mask=None):
        
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )

        rgb_feats = rgb_feats_pyramid[0]
        pcd = pcd_pyramid[0]

        if self.use_mask:
            context_feats, context, mask_idx = self.encoder.mask_out_features_pcd(mask, rgb_feats, pcd, n_min=0, n_max=1024)
        else:
            # Keep only low-res scale
            context_feats = einops.rearrange(
                rgb_feats_pyramid[0],
                "b ncam c h w -> b (ncam h w) c"
            )
            context = pcd_pyramid[0]

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )
    
        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pcd = self.encoder.run_fps(
            context_feats,
            context
        )

        return (
            context_feats, context,  # contextualized visual features
            adaln_gripper_feats, curr_gripper, # gripper history features
            fps_feats, fps_pcd  # sampled visual features
        )


    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            context_feats,
            context,
            adaln_gripper_feats,
            curr_gripper_pose,
            fps_feats,
            fps_pcd
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            adaln_gripper_feats=adaln_gripper_feats,
            curr_gripper_pose=curr_gripper_pose,
            fps_feats=fps_feats,
            fps_pcd=fps_pcd
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        # Random trajectory, conditioned on start-end
        B, L, _, _ = condition_data.size()
        device = condition_data.device
        # Iterative denoising
        with torch.no_grad():
            dt = 1.0 / self.n_steps
            r0, p0 = self.generate_random_initial_pose(batch=B, trj_steps=L)
            r0, p0 = r0.to(device), p0.to(device)
            rt, pt = r0, p0
            for s in range(0, self.n_steps):
                time = s*dt*torch.ones_like(pt[:, 0, 0], device=device)
                xt = se3_from_rot_pos(rt, pt)
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
        rgb_obs,
        pcd_obs,
        curr_gripper,
        mask_obs=None
    ):
        
        # Convert rotation parametrization
        curr_gripper, _ = self.convert_rot(curr_gripper)

        if self._relative:
            curr_gripper_abs = curr_gripper.clone()
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)


        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, curr_gripper, mask_obs
        )

        # Condition on start-end pose
        B, nhist, D1, D2 = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D1, D2),
            device=pcd_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        output = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs,
        )

        trajectory = output['trajectory']   
        gripper_openess = output['gripper_openess']         

        # Normalize quaternion
        if self._rotation_parametrization == 'quat':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])

        if self._relative:
            trajectory = self.convert2abs(trajectory, curr_gripper_abs)
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory, res=gripper_openess)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.size(-1) > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        output['trajectory'] = trajectory

        return output

    def normalize_pos(self, pos):
        if self.gripper_loc_bounds is None:
            return pos * self.scaling_factor
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        if self.gripper_loc_bounds is None:
            return pos / self.scaling_factor
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

    def convert2rel(self, pcd, curr_gripper, trajectory=None):
        """Convert coordinate system relaative to current gripper."""
        trans, rot = se3_inverse(curr_gripper[:,-1, :3, 3], curr_gripper[:,-1, :3, :3])
        inv_pose = se3_from_rot_pos(rot, trans)

        bs = trans.shape[0]       
        pcd = pcd.clone() 
        pcd = einsum('bmn,bvnhw->bvmhw', rot, pcd) + trans.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper = einsum('bmn,bhnk->bhmk', inv_pose, curr_gripper)
        if trajectory is not None:
            trajectory = trajectory.clone()
            trajectory = einsum('bmn,blnk->blmk', inv_pose, trajectory)
            return pcd, curr_gripper, trajectory
        return pcd, curr_gripper
    
    def convert2abs(self, trajectory, curr_gripper, pcd=None):
        pose = se3_from_rot_pos(curr_gripper[:, -1, :3, :3], curr_gripper[:, -1, :3, 3])
        trajectory = einsum('bmn,blnk->blmk', pose, trajectory)
        if pcd is not None:
            bs = pcd.shape[0]
            pcd = einsum('bmn,bkn->bkm', pose[:, -1, :3, :3], pcd) + pose[:, -1, :3, 3].view(bs, 1, 3)
            curr_gripper = einsum('bmn,bhnk->bhmk', pose, curr_gripper)
            return trajectory, curr_gripper, pcd
        return trajectory
    
    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        curr_gripper,
        run_inference=False,
        mask_obs=None,
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
        # Normalize all pos
        if gt_trajectory is not None:
            gt_trajectory = gt_trajectory.clone()
            gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

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
                curr_gripper,
                mask_obs
            )
        # Convert rotation parametrization
        gt_trajectory, _ = self.convert_rot(gt_trajectory)
        curr_gripper, _ = self.convert_rot(curr_gripper)
        
        if self._relative:
            pcd_obs, curr_gripper, gt_trajectory = self.convert2rel(pcd_obs, curr_gripper, gt_trajectory)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, curr_gripper, mask_obs
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
        trajectory_t = se3_from_rot_pos(rt, pt)

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
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trajectory_mask = torch.zeros(1, self.nhorizon, device=obs_dict['rgb'].device)
        output = self.forward(
            gt_trajectory=None,
            trajectory_mask=trajectory_mask,
            rgb_obs=obs_dict['rgb'],
            pcd_obs=obs_dict['pcd'],
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
            mask_obs=obs_dict.get('mask', None)
        )
        rlbench_action = output['trajectory'].clone()

        if rlbench_action.shape[-1] > 7:
            rlbench_action[..., 7] = rlbench_action[..., 7] > 0.5
            
        action = create_robomimic_from_rlbench_action(rlbench_action, quaternion_format = self._quaternion_format)
        result = {
            'rlbench_action' : rlbench_action,
            'action': action,
        }

        return result
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            gt_trajectory=batch['action']['gt_trajectory'],
            trajectory_mask=None,
            rgb_obs=batch['obs']['rgb'],
            pcd_obs=batch['obs']['pcd'],
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
        gt_act_gr = gt_trajectory[..., 7]

        out = self.predict_action(batch['obs'])
        action = out['action']
        pred_act_p = action['act_p']
        pred_act_r = action['act_r']
        pred_act_gr = out['rlbench_action'][:,:,7]

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
                 rotation_parametrization='so3',
                 nhist=3,
                 nhorizon=1,
                 causal_attn=True):
        super().__init__()
        if 'so3' in rotation_parametrization:
            rotation_out_dim = 3
            rotation_in_dim = 9
        else:
            raise ValueError(f"Unknown rotation parametrization: {rotation_parametrization}")

        # Encoders
        self.trajectory_embeddings = nn.Embedding(nhorizon, embedding_dim)
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

        head_dim = embedding_dim // num_attn_heads
        assert head_dim * num_attn_heads == embedding_dim, "embed_dim must be divisible by num_heads"

        # Estimate attends to context (no subsampling)
        self.cross_attn = InvariantPointTransformer(
            embedding_dim, 2, num_attn_heads, head_dim, kv_dim=embedding_dim, use_adaln=True,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3
        )


        self.self_attn = InvariantPointTransformer(
            embedding_dim, 4, num_attn_heads, head_dim, kv_dim=None, use_adaln=True,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3)


        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = InvariantPointTransformer(
            embedding_dim, 2, num_attn_heads, head_dim, kv_dim=None, use_adaln=True,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3)

        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_out_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = InvariantPointTransformer(
            embedding_dim, 2, num_attn_heads, head_dim, kv_dim=None, use_adaln=True,
            dropout=0.0, attention_module=InvariantPointAttention, point_dim=3)
        

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

        self.causal_attn = causal_attn

    def forward(self, trajectory, timestep,
                context_feats, context, adaln_gripper_feats,
                curr_gripper_pose, fps_feats, fps_pcd):
        """
        Arguments:
            trajectory: (B, trajectory_length, 4, 4)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, 3)
            adaln_gripper_feats: (B, nhist, F)
            curr_gripper_pose: (B, nhist, 4, 4)
        """
        # Trajectory features
        traj_feats = self.trajectory_embeddings.weight.unsqueeze(0).repeat(trajectory.shape[0], 1, 1)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        traj_feats = traj_feats + traj_time_pos        

        # Predict position, rotation, opening
        output = self.prediction_head(
            trajectory, traj_feats,
            context, context_feats,
            timestep, adaln_gripper_feats,
            curr_gripper_pose=curr_gripper_pose,
            fps_feats=fps_feats, fps_pcd=fps_pcd
        )

        pos_pred, rot_pred, openess_pred = output['pred']
        output['pred'] = [torch.cat((pos_pred, rot_pred, openess_pred), -1)]
        return output
    
    def prediction_head(self,
                        trajectory_pose, trajectory_features,
                        pcd, pcd_features,
                        timesteps, curr_gripper_features,
                        curr_gripper_pose, fps_feats, fps_pcd):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pose: A tensor of shape (B, N, 4, 4)
            gripper_features: A tensor of shape (B, N, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (B, N, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (B, M, F)
            curr_gripper: A tensor of shape (B, M, 4, 4)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        pcd_rotations = torch.eye(3).reshape(1,1,3,3).repeat(pcd.size(0), pcd.size(1), 1, 1).to(pcd.device)
        pcd_translations = pcd

        time_feats = self.time_emb(timesteps)
        trajectory_features = trajectory_features + time_feats[:, None, :]
        curr_gripper_features = curr_gripper_features + time_feats[:, None, :]
        pcd_features = pcd_features + time_feats[:, None, :]

        poses_x = {'rotations': trajectory_pose[..., :3, :3], 
                   'translations': trajectory_pose[..., :3, 3], 'types': 'se3'}
        poses_z = {'rotations': torch.cat([curr_gripper_pose[..., :3, :3], pcd_rotations], 1),
                   'translations': torch.cat([curr_gripper_pose[..., :3, 3], pcd_translations], 1), 'types': 'se3'}
        point_mask_z = torch.cat([torch.zeros_like(curr_gripper_pose[..., 0, 0]).bool(), torch.ones_like(pcd[..., 0]).bool()], 1)
        context = torch.cat([curr_gripper_features, pcd_features], 1)
        features = self.cross_attn(x=trajectory_features, poses_x=poses_x, z=context, poses_z=poses_z, point_mask_z=point_mask_z, diff_ts=time_embs)

        # Self-attention
        fps_rotations = torch.eye(3).reshape(1,1,3,3).repeat(fps_pcd.size(0), fps_pcd.size(1), 1, 1).to(fps_pcd.device)

        poses_x = {'rotations': torch.cat([poses_x['rotations'], curr_gripper_pose[..., :3, :3], fps_rotations], 1),
                     'translations': torch.cat([poses_x['translations'], curr_gripper_pose[..., :3, 3], fps_pcd], 1), 'types': 'se3'}
        point_mask_x = torch.cat([torch.zeros_like(trajectory_pose[..., 0, 0]).bool(), torch.zeros_like(curr_gripper_pose[..., 0, 0]).bool(), torch.ones_like(fps_pcd[..., 0]).bool()], 1)
        features = torch.cat([features, curr_gripper_features, fps_feats], 1)
        num_trajectory = trajectory_features.shape[1]
        num_observations = fps_feats.shape[1] + curr_gripper_features.shape[1]
        if self.causal_attn:
            mask = torch.ones(num_trajectory + num_observations, num_trajectory + num_observations)
            mask[:num_trajectory, :num_trajectory] = torch.tril(mask[:num_trajectory, :num_trajectory])
            mask[num_trajectory:, :num_trajectory] = torch.zeros_like(mask[num_trajectory:, :num_trajectory])
            mask = mask.to(features.device).bool()
        else:
            mask = None

        features = self.self_attn(x=features, poses_x=poses_x, point_mask_x=point_mask_x, diff_ts=time_embs, mask=mask)

        # Rotation head
        rotation = self.predict_rot(
            features, poses_x, point_mask_x, num_trajectory, time_embs, mask
        )

        # Position head
        position, position_features = self.predict_pos(
            features, poses_x, point_mask_x, num_trajectory, time_embs, mask
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        output = {
            'pred': [position, rotation, openess],
        }

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

        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, poses, point_mask, num_trajectory, time_embs, mask=None):
        position_features = self.position_self_attn(x=features, poses_x=poses, point_mask_x=point_mask, diff_ts=time_embs, mask=mask)        
        position_features = self.position_proj(position_features)  # (B, N, C)
        position_features = position_features[:, :num_trajectory]
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, poses, point_mask, num_trajectory, time_embs, mask=None):
        rotation_features = self.rotation_self_attn(x=features, poses_x=poses, point_mask_x=point_mask, diff_ts=time_embs, mask=mask)        
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation_features = rotation_features[:, :num_trajectory]
        rotation = self.rotation_predictor(rotation_features)
        return rotation


with torch.no_grad():
    def test():
        from diffusion_policy.common.pytorch_util import dict_apply
        from diffusion_policy.model.common.se3_util import random_se3
        from diffusion_policy.common.rlbench_util import se3_to_gripper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        horizon = 1
        nhist = 3
        nkeypoints = 10

        model = DiffuserActor(
            embedding_dim=192,
            fps_subsampling_factor=5,
            gripper_loc_bounds=[[-1, -1, -1], [1, 1, 1]],
            rotation_parametrization='so3',
            quaternion_format='xyzw',
            diffusion_timesteps=100,
            nhist=nhist,
            nhorizon=horizon,
            nkeypoints=nkeypoints,
        )

        model.to(device)

        batch = {
            'action': {
                'gt_trajectory': torch.cat([torch.randn(1, 1, 3), normalise_quat(torch.randn(1, 1, 4)), torch.randn(1, 1, 1)], -1),
            },
            'obs': {
                'low_dim_pcd': torch.randn(1, 10, 3),
                'curr_gripper': torch.cat([torch.randn(1, nhist, 3), normalise_quat(torch.randn(1, nhist, 4)), torch.randn(1, nhist, 2)], -1)
            }
        }

        batch = dict_apply(batch, lambda x: x.to(device))


        # rotate global frame
        def convert2rel(pcd, curr_gripper, trajectory, H):
            trans, rot = se3_inverse(H[:, :3, 3], H[:, :3, :3])
            inv_pose = se3_from_rot_pos(rot, trans)

            bs = trans.shape[0]       
            pcd = pcd.clone() 
            pcd = einsum('bmn,bkn->bkm', rot, pcd) + trans.view(bs, 1, 3)
            curr_gripper = curr_gripper.clone()
            curr_gripper = einsum('bmn,bhnk->bhmk', inv_pose, curr_gripper)
            trajectory = trajectory.clone()
            trajectory = einsum('bmn,blnk->blmk', inv_pose, trajectory)
            return pcd, curr_gripper, trajectory
            
        def convert2abs(trajectory, curr_gripper, pcd, H):
            pose = se3_from_rot_pos(H[:, :3, :3], H[:, :3, 3])
            trajectory = einsum('bmn,blnk->blmk', pose, trajectory)
            bs = pcd.shape[0]
            pcd = einsum('bmn,bkn->bkm', pose[:, :3, :3], pcd) + pose[:, :3, 3].view(bs, 1, 3)
            curr_gripper = einsum('bmn,bhnk->bhmk', pose, curr_gripper)
            return trajectory, curr_gripper, pcd

        H = random_se3(1).to(device)
        pcd0, curr_gripper0, trajectory0 = batch['obs']['low_dim_pcd'], batch['obs']['curr_gripper'], batch['action']['gt_trajectory']
        curr_gripper, _ = model.convert_rot(curr_gripper0)
        trajectory, _ = model.convert_rot(trajectory0)
        pcd = pcd0.clone()
        pcd_, curr_gripper_, trajectory_ = convert2rel(pcd, curr_gripper, trajectory, H)
        batch_ = {
            'action': {
                'gt_trajectory': se3_to_gripper(trajectory_, res=trajectory0[..., 7:]),
            },
            'obs': {
                'low_dim_pcd': pcd_,
                'curr_gripper': se3_to_gripper(curr_gripper_, res=curr_gripper0[..., 7:])
            }
        }
        trajectory_, curr_gripper_, pcd_ = convert2abs(trajectory_, curr_gripper_, pcd_, H)

        assert torch.allclose(pcd, pcd_, rtol=1e-5, atol=1e-5), "Frame translation/rotation failed"
        assert torch.allclose(curr_gripper, curr_gripper_, rtol=1e-5, atol=1e-5), "Frame translation/rotation failed"
        assert torch.allclose(trajectory, trajectory_, rtol=1e-5, atol=1e-5), "Frame translation/rotation failed"

        torch.random.manual_seed(0)
        loss1 = model.compute_loss(batch)
        torch.random.manual_seed(0)
        loss2 = model.compute_loss(batch_)
        assert torch.allclose(loss1, loss2, rtol=1e-5, atol=1e-5), "Frame translation/rotation failed"

        torch.random.manual_seed(0)
        act1 = model.predict_action(batch['obs'])
        torch.random.manual_seed(0)
        act2 = model.predict_action(batch_['obs'])
        assert torch.allclose(act1['rlbench_action'], act2['rlbench_action'], rtol=1e-5, atol=1e-5), "Frame translation/rotation failed"

        torch.random.manual_seed(0)
        out1 = model.evaluate(batch)
        torch.random.manual_seed(0)
        out2 = model.evaluate(batch_)

        print("Success")

if __name__ == "__main__":
    test()
    print("Test passed")