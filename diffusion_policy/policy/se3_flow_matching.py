import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.common.so3_util import log_map, se3_inverse
from diffusion_policy.common.se3_util import se3_from_rot_pos
from diffusion_policy.common.rotation_utils import normalise_quat
from diffusion_policy.model.flow_matching.flow_matching_models import SE3LinearAttractorFlow

from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer

from diffusion_policy.model.obs_encoders.se3_grasp_pcd_encoder import SE3GraspPointCloudSuperEncoder
from diffusion_policy.model.obs_encoders.feature_pcd_encoder import FeaturePCDEncoder
from diffusion_policy.model.flow_matching.se3_grasp_vector_field import SE3GraspVectorField

from torch import einsum

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from diffusion_policy.common.so3_util import log_map
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import Dict


class SE3FlowMatching(BaseImagePolicy):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 n_points_out=100,
                 scaling_factor=3.0,
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 nhorizon=16,
                 t_switch=0.75,
                 relative=False,
                 causal_attn=True,
                 gripper_loc_bounds=None,
                 workspace_bounds=None,
                 max_pcd_points=None,
                 feature_res="res2",
                 pcd_self_attn=False,
                 ):
        super().__init__()
        self._quaternion_format = quaternion_format
        self.feature_pcd_encoder = FeaturePCDEncoder(
            backbone=backbone,
            feature_res=feature_res
        )
        encoder = SE3GraspPointCloudSuperEncoder(
            dim_features=embedding_dim,
            depth=3,
            nheads=4,
            n_steps_inf=50,
            n_points_out=n_points_out,
            nhist=nhist,
            dim_pcd_features=self.feature_pcd_encoder.out_dim
        )
        decoder = URSATransformer(d_model=embedding_dim, nhead=4, num_layers=2)
        self.model = SE3GraspVectorField(
            encoder=encoder, 
            decoder=decoder, 
            latent_dim=embedding_dim)

        self.n_steps = diffusion_timesteps
        self.nhorizon = nhorizon
        
        ## Flow Model ##
        self.t_switch = t_switch
        self.scaling_factor = torch.tensor(scaling_factor)
        self.flow = SE3LinearAttractorFlow(t_switch=self.t_switch)

        self._relative = relative
        self.pcd_self_attn = pcd_self_attn
        if gripper_loc_bounds is not None:
            self.register_buffer("gripper_loc_bounds", torch.tensor(gripper_loc_bounds))
        else:
            self.gripper_loc_bounds = None
        if workspace_bounds is not None:
            self.register_buffer("workspace_bounds", torch.tensor(workspace_bounds))
        else:
            self.workspace_bounds = None
        self.max_pcd_points = max_pcd_points

    def crop_to_workspace(self, pcd, feats, workspace_bounds):
        """
        Returns the indices of points within the specified workspace bounds, ensuring each batch has an equal number of points.

        Parameters:
        - pcd: A tensor of shape (B, N, 3) representing the point cloud.
        - feats: A tensor of shape (B, N, F) representing the features.
        - workspace_bounds: A list or tensor of shape (2, 3) specifying the min and max bounds for x, y, z.

        Returns:
        - trimmed_pcd: A list of tensors, where each tensor contains the selected points for a batch.
        - trimmed_feats: A list of tensors, where each tensor contains the feature values corresponding to the selected points for a batch.
        """
        batch_size = pcd.shape[0]
        batch_indices = []

        for b in range(batch_size):
            mask = torch.ones(pcd[b].shape[0], dtype=torch.bool, device=pcd.device)
            for i in range(3):
                mask = torch.logical_and(mask, pcd[b, :, i] > workspace_bounds[0][i])
                mask = torch.logical_and(mask, pcd[b, :, i] < workspace_bounds[1][i])

            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)  # Get the indices where mask is True
            # Randomly sample points if there are more than max_pcd_points
            if len(indices) > self.max_pcd_points:
                indices = indices[torch.randperm(len(indices))[:self.max_pcd_points]]
            batch_indices.append(indices)

        # Extract the corresponding points and RGB values
        cropped_pcd = [pcd[b, indices, :] for b, indices in enumerate(batch_indices)]
        cropped_feats = [feats[b, indices, :] for b, indices in enumerate(batch_indices)]

        cropped_pcd = torch.stack(cropped_pcd)
        cropped_feats = torch.stack(cropped_feats)

        return cropped_pcd, cropped_feats 

        
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
        signal = signal.clone()
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        # The following code expects wxyz quaternion format!
        if self._quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
        rot = quaternion_to_matrix(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        H = se3_from_rot_pos(rot, signal[..., :3])
        return H, res

    def unconvert_rot(self, H, res=None):
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
        trans, rot = se3_inverse(self.relative_frame[:, :3, 3], self.relative_frame[:, :3, :3])
        inv_pose = se3_from_rot_pos(rot, trans)

        bs = trans.shape[0]       
        pcd = pcd.clone() 
        pcd = einsum('bmn,bln->bln', rot, pcd) + trans.view(bs, 1, 3)
        curr_gripper = curr_gripper.clone()
        curr_gripper = einsum('bmn,bhnk->bhmk', inv_pose, curr_gripper)
        if trajectory is not None:
            trajectory = trajectory.clone()
            trajectory = einsum('bmn,blnk->blmk', inv_pose, trajectory)
        return pcd, curr_gripper, trajectory
    
    def convert2abs(self, trajectory, pcd=None):
        trajectory = einsum('bmn,blnk->blmk', self.relative_frame, trajectory)
        if pcd is not None:
            bs = pcd.shape[0]
            pcd = einsum('bmn,bkn->bkm', self.relative_frame[:, :3, :3], pcd) + self.relative_frame[:, :3, 3].view(bs, 1, 3)
            return trajectory, pcd
        return trajectory

    def sample(self, fixed_inputs):
        B = fixed_inputs["obs"]["pcd"].shape[0]
        device = fixed_inputs["obs"]["pcd"].device
        # Iterative denoising
        with torch.no_grad():
            dt = 1.0 / self.n_steps
            r0, p0 = self.flow.generate_random_initial_pose(batch=B, trj_steps=1)
            r0, p0 = r0.to(device), p0.to(device)
            rt, pt = r0, p0
            for s in range(0, self.n_steps):
                time = s*dt*torch.ones_like(pt[:, 0, 0], device=device)
                xt = se3_from_rot_pos(rt, pt)
                out, gr = self.model.forward_act({
                    'act': xt,
                    'time': time
                })
                dp, dr = out[...,:3], out[...,3:6]   
                rt, pt = self.flow.step(rt, pt, dr, dp, dt, time=s*dt)


        trajectory = se3_from_rot_pos(rt, pt)

        if self._relative:
            trajectory = self.convert2abs(trajectory)
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory, res=gr > 0.5)
        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)

        output = dict()
        output['trajectory'] = trajectory
        output['gripper_openess'] = gr

        return output
    

    
    def create_obs_dict(self, pcd, curr_gripper, feature_obs):
        obs = dict()
        obs['pcd'] = pcd
        obs['current_gripper'] = curr_gripper
        obs['pcd_features'] = feature_obs
        return obs
   
    def forward(
        self,
        gt_trajectory,
        rgb_obs,
        pcd_obs,
        curr_gripper,
        run_inference=False,
        feature_obs=None
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
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
            if self.workspace_bounds is not None:
                pcd_obs, feature_obs = self.crop_to_workspace(pcd_obs, feature_obs, self.workspace_bounds)       

        if gt_trajectory is not None:
            gt_trajectory = self.normalize_pos(gt_trajectory)
        pcd_obs = self.normalize_pos(pcd_obs)
        curr_gripper = self.normalize_pos(curr_gripper)

        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:8]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # Convert rotation parametrization
        curr_gripper, _ = self.convert_rot(curr_gripper)
        if gt_trajectory is not None:
            gt_trajectory, _ = self.convert_rot(gt_trajectory)

        if self._relative:
            self.relative_frame = se3_from_rot_pos(curr_gripper[:, -1, :3, :3], curr_gripper[:, -1, :3, 3])
            pcd_obs, curr_gripper, gt_trajectory = self.convert2rel(pcd_obs, curr_gripper, gt_trajectory)

        obs = self.create_obs_dict(pcd_obs, curr_gripper, feature_obs)

        if run_inference:
            return self.sample({'obs': obs})
            
        # Prepare inputs
        p1 = gt_trajectory[:, :, :3, 3]
        r1 = gt_trajectory[:, :, :3, :3]

        # Add noise to the clean trajectories
        r0, p0 = self.flow.generate_random_initial_pose(batch=gt_trajectory.shape[0], trj_steps=gt_trajectory.shape[1])
        r0, p0 = r0.to(gt_trajectory.device), p0.to(gt_trajectory.device)
        timesteps = torch.rand(gt_trajectory.shape[0], device=gt_trajectory.device)
        rt, pt = self.flow.flow_at_t(r0, p0, r1, p1, timesteps)
        dr, dp = self.flow.vector_field_at_t(r1,p1,rt,pt,timesteps)
        
        # Predict the noise residual
        trajectory_t = se3_from_rot_pos(rt, pt)

        obs_x, obs_f = self.model.encode_obs({'obs': obs})
        self.model.set_context(obs_x, obs_f)
        # Predict the noise residual
        input_data = {'obs': obs, 'act': trajectory_t, 'time': timesteps}
        d_act, openess = self.model.forward_act(input_data)

        # Compute loss
        loss = 30 * F.mse_loss(dp, d_act[..., :3], reduction='mean') + 10 * F.mse_loss(dr, d_act[..., 3:6], reduction='mean')
        if torch.numel(gt_openess) > 0:
            loss += F.binary_cross_entropy(openess, gt_openess)
        return loss
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(
            gt_trajectory=None,
            rgb_obs=obs_dict.get('rgb', None),
            pcd_obs=obs_dict['pcd'],
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
            feature_obs=obs_dict.get('clip_features', None)
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            gt_trajectory=batch['action']['gt_trajectory'],
            rgb_obs=batch['obs'].get('rgb', None),
            pcd_obs=batch['obs']['pcd'],
            curr_gripper=batch['obs']['curr_gripper'],
            run_inference=False,
            feature_obs=batch['obs'].get('clip_features', None)
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
        pred_act, pred_act_gr = self.convert_rot(trajectory)
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
    
with torch.no_grad():
    def test():
        from diffusion_policy.common.pytorch_util import dict_apply
        from diffusion_policy.dataset.rlbench_dataset import RLBenchDataset
        import os

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        horizon = 1
        nhist = 3

        model = SE3FlowMatching(
            embedding_dim=192,
            n_points_out=100,
            gripper_loc_bounds=[[-1, -1, -1], [1, 1, 1]],
            quaternion_format='xyzw',
            diffusion_timesteps=100,
            nhist=nhist,
            nhorizon=horizon,
        )

        model.to(device)

        dataset = RLBenchDataset(
            dataset_path=os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data/peract.zarr'),
            cameras=['left_shoulder', 'right_shoulder', 'wrist', 'front'],
            task_name='open_drawer',
            use_rgb=True,
            use_pcd=True,
            use_mask=False,
            use_features=False,
            n_obs_steps=3,
            n_episodes=-1,
            image_rescale=(1.0, 1.0),
            cache_size=0,
            use_precomputed_features=False
        )


        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1
        )
        batch = next(iter(dataloader))
        batch = dict_apply(batch, lambda x: x.to(device))
        loss = model.compute_loss(batch)
        print("Success")

if __name__ == "__main__":
    test()
    print("Test passed")