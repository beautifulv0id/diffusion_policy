import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.common.so3_util import log_map, se3_inverse
from diffusion_policy.common.se3_util import se3_from_rot_pos
from diffusion_policy.common.rotation_utils import normalise_quat
from geo3dattn.policy.se3_flowmatching.common.se3_flowmatching import RectifiedLinearFlow

from geo3dattn.model.ursa_transformer.ursa_transformer import URSATransformer
from geo3dattn.model.direct_transformer.direct_transformer import DirectTransformer

from diffusion_policy.model.obs_encoders.se3_grasp_pcd_encoder import SE3GraspPointCloudSuperEncoder
from diffusion_policy.model.obs_encoders.feature_pcd_encoder import FeaturePCDEncoder
from diffusion_policy.model.flow_matching.se3_grasp_vector_field import SE3GraspVectorField
from diffusion_policy.model.common.workspace_cropping import crop_to_workspace

from einops import reduce
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
                 relative=False,
                 causal_attn=True,
                 gripper_loc_bounds=None,
                 workspace_bounds=None,
                 crop_workspace=True,
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
            nheads=8,
            n_steps_inf=diffusion_timesteps,
            n_points_out=n_points_out,
            nhist=nhist,
            dim_pcd_features=self.feature_pcd_encoder.out_dim
        )
        decoder = DirectTransformer(d_model=embedding_dim, nhead=8, num_layers=4)
        self.model = SE3GraspVectorField(
            encoder=encoder, 
            decoder=decoder, 
            latent_dim=embedding_dim)

        self.nhorizon = nhorizon
        
        ## Flow Model ##
        self.scaling_factor = torch.tensor(scaling_factor, requires_grad=False)
        self.flow = RectifiedLinearFlow(n_action_steps=1, num_steps=diffusion_timesteps)

        self._relative = relative
        self.pcd_self_attn = pcd_self_attn
        if gripper_loc_bounds is not None:
            self.register_buffer("gripper_loc_bounds", torch.tensor(gripper_loc_bounds, requires_grad=False))
        else:
            self.gripper_loc_bounds = None
        if workspace_bounds is not None and crop_workspace:
            self.register_buffer("workspace_bounds", torch.tensor(workspace_bounds, requires_grad=False))
        else:
            self.workspace_bounds = None
        self.max_pcd_points = max_pcd_points


    # ========= utils  ============
    def vec_to_pose(self, vec):
        p, r = self.flow._vector_to_pose(vec)
        H = torch.eye(4)[None, None, ...].repeat(vec.shape[0], vec.shape[1], 1, 1).to(vec.device)
        H[:, :, :3, -1] = p
        H[:, :, :3, :3] = r
        return H

    def pose_to_vec(self, H):
        p, r = H[:, :, :3, -1], H[:, :, :3, :3]
        vec = self.flow._pose_to_vector(p, r)
        return vec

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

    def convert2rel(self, trajectory):
        """Convert coordinate system relaative to current gripper."""
        trans, rot = se3_inverse(self.relative_frame[:, :3, 3], self.relative_frame[:, :3, :3])
        inv_pose = se3_from_rot_pos(rot, trans)
        trajectory = einsum('bmn,blnk->blmk', inv_pose, trajectory)
        return trajectory
    
    def convert2abs(self, trajectory):
        trajectory = einsum('bmn,blnk->blmk', self.relative_frame, trajectory)
        return trajectory
    
    def set_mean_std(self, mean, std):
        mean = self.normalize_pos(mean)
        std = self.normalize_pos(std)
        self.flow.set_mean_std(mean, std)

    def sample(self, obs):
        B = obs["pcd"].shape[0]
        self.model.set_context(*self.model.encode_obs(obs))
        # Iterative denoising
        with torch.no_grad():
            at = self.flow.generate_random_initial_pose(B)
            for s in range(0, self.flow.num_steps):
                step = s * torch.ones_like(at[:, 0, 0])
                at_H = self.vec_to_pose(at)
                d_act, gripper_open = self.model.forward_act({
                    'act': at_H,
                    'time':step})
                at = self.flow.step(at, d_act, s)

        trajectory = self.vec_to_pose(at)

        return trajectory, gripper_open
       
    def forward(
        self,
        gt_trajectory,
        rgb_obs,
        pcd_obs,
        curr_gripper,
        instruction=None,  # note: not used
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
        # Compute rgb features
        if feature_obs is None:
            feature_obs, pcd_obs = self.feature_pcd_encoder(rgb_obs, pcd_obs)
            if self.workspace_bounds is not None:
                pcd_obs, feature_obs = crop_to_workspace(pcd_obs, feature_obs, self.workspace_bounds, self.max_pcd_points)       

        # Normalize position
        if gt_trajectory is not None:
            gt_trajectory = self.normalize_pos(gt_trajectory)
        pcd_obs = self.normalize_pos(pcd_obs)
        curr_gripper = self.normalize_pos(curr_gripper)
        curr_gripper = curr_gripper[..., :7]

        # Convert rotation parametrization
        curr_gripper, _ = self.convert_rot(curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:8]
            gt_trajectory, _ = self.convert_rot(gt_trajectory)

        # Convert to relative frame of gripper
        if self._relative:
            self.relative_frame = se3_from_rot_pos(curr_gripper[:, -1, :3, :3], curr_gripper[:, -1, :3, 3]).detach()
            if gt_trajectory is not None:
                gt_trajectory = self.convert2rel(gt_trajectory)

        # Create observation dictionary
        obs = {
            'pcd': pcd_obs,
            'current_gripper': curr_gripper,
            'pcd_features': feature_obs
        }

        if run_inference:
            return self.sample(obs)
            
        # Prepare inputs
        batch_size = pcd_obs.shape[0]
        device, dtype = pcd_obs.device, pcd_obs.dtype
        act_vector = self.flow._pose_to_vector(gt_trajectory[...,:3, -1], gt_trajectory[...,:3, :3])

        # 2. Compute Flow Matching Variables
        a1 = act_vector
        a0 = self.flow.generate_random_initial_pose(batch_size)
        time = torch.randint(0, self.flow.num_steps, (batch_size,)).to(device=device, dtype=dtype)

        at = self.flow.flow_at_t(a0, a1, time)
        target = self.flow.vector_field_at_t(a0, a1, at, time)

        ## 3. Set Context
        self.model.set_context(*self.model.encode_obs(obs))

        # Predict the noise residual
        at_pose = self.vec_to_pose(at)
        input_data = {'obs': obs, 'act': at_pose, 'time': time}
        d_act, openess = self.model.forward_act(input_data)

        # Compute loss
        loss = (
                30 * F.l1_loss(d_act[...,:3], target[...,:3], reduction='mean')
                + 10 * F.l1_loss(d_act[..., 3:6], target[..., 3:6], reduction='mean')
        )
        if torch.numel(gt_openess) > 0:
            loss += F.binary_cross_entropy(openess, gt_openess)
        return loss
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trajectory, gripper_open = self.forward(
            gt_trajectory=None,
            rgb_obs=obs_dict.get('rgb', None),
            pcd_obs=obs_dict['pcd'],
            curr_gripper=obs_dict['curr_gripper'],
            run_inference=True,
            feature_obs=obs_dict.get('clip_features', None)
        )
    
        if self._relative:
            trajectory = self.convert2abs(trajectory)
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory, res=gripper_open > 0.5)
        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)

        output = dict()
        output['trajectory'] = trajectory
        output['gripper_openess'] = gripper_open

        return output

    
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