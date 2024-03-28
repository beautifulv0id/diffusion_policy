from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.flow_matching.feat_pcloud_naive_policy import NaivePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.transformer_lowdim_obs_relative_encoder import TransformerHybridObsRelativeEncoder
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.common_utils import action_from_trajectory_gripper_open_ignore_collision
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.se3_diffusion_util import marginal_prob_std, sample_from_se3_gaussian, se3_score_normal, step
import theseus as th
from theseus.geometry.so3 import SO3
from scipy.spatial.transform import Rotation 
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

class FlowMatchingUnetLowDimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: TransformerHybridObsRelativeEncoder,
            horizon : int, 
            n_action_steps : int, 
            n_obs_steps : int,
            num_inference_steps=1000,
            delta_t=0.01,
            gripper_loc_bounds=None,
            relative_position=True,
            relative_rotation=True,
            noise_aug_std=0.1,
            velocity_scale=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 2
        # TODO: make this more general
        action_dim = 12
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = 16 * horizon


        model = NaivePolicy(
            dim=input_dim,
            output_size=6,
            context_dim=obs_feature_dim,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.gripper_state_ignore_collision_predictor = nn.Sequential(
            nn.Linear(obs_feature_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 2*horizon),
        )

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)
        self.relative_position = relative_position
        self.relative_rotation = relative_rotation
        self.kwargs = kwargs
        self.num_inference_steps = num_inference_steps
        self.dt = delta_t
        self.velocity_scale = velocity_scale
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")
        if noise_aug_std > 0:
            self.noise_aug_std = noise_aug_std
            self.data_augmentation = True
        else:
            self.data_augmentation = False

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def state_dict(self):
        super_dict = super().state_dict()
        super_dict['relative_position'] = self.relative_position
        super_dict['relative_rotation'] = self.relative_rotation
        return super_dict
    
    def load_state_dict(self, state_dict):
        self.relative_position = state_dict.pop('relative_position', self.relative_position)
        self.relative_rotation = state_dict.pop('relative_rotation', self.relative_rotation)
        super().load_state_dict(state_dict)
    
    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
    
    def to_rel_trajectory(self, traj, agent_pos):
        """
        Args:
            traj (torch.Tensor): (B, t, Da)
            agent_pos (torch.Tensor): (B, Da)

        Returns:
            torch.Tensor: (B, t, Da)
        """
        traj = traj - agent_pos[:, None, :]
        return traj

    def to_abs_trajectory(self, traj, agent_pos):
        """
        Args:
            traj (torch.Tensor): (B, t, Da)
            agent_pos (torch.Tensor): (B, Da)

        Returns:    
            torch.Tensor: (B, t, Da)
        """
        traj = traj + agent_pos[:, None, :]
        return traj
        
    def normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        this_obs = dict_apply(obs_dict, lambda x: x.clone())
        this_obs['agent_pose'][:,:,:3,3] = self.normalize_pos(this_obs['agent_pose'][:,:,:3,3])
        this_obs['keypoint_pcd'] = self.normalize_pos(this_obs['keypoint_pcd'])
        return obs_dict

    def normalize_trajectory(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clone()
        action[:,:,:3,3] = self.normalize_pos(action[:,:,:3,3])
        return action
    
    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clone()
        action[:,:,:3,3] = self.unnormalize_pos(action[:,:,:3,3])
        return action
    
    def toHomogeneous(self, x):
        return torch.cat([x, torch.ones_like(x[...,0:1])], dim=-1)
    
    def convert2rel(self, x_curr, R_curr, obs, action=None):
        this_obs = dict_apply(obs, lambda x: x.clone())
        H = torch.eye(4, device=x_curr.device, dtype=x_curr.dtype).reshape(1,4,4).repeat(x_curr.shape[0], 1, 1)
        if self.relative_position:
            H[:,:3,3] = -x_curr
        if self.relative_rotation:
            H[:,:3,3] = torch.einsum('bmn,bn->bm', R_curr.transpose(-1,-2), H[:,:3,3])
            H[:,:3,:3] = R_curr.transpose(-1,-2) 
        this_obs['keypoint_pcd'] = torch.einsum('bmn,bhkn->bhkm', H, self.toHomogeneous(this_obs['keypoint_pcd']))[..., :3]
        this_obs['agent_pose'] = torch.einsum('bmn,bhnk->bhmk', H, this_obs['agent_pose'])
        if action is not None:
            this_action = torch.einsum('bmn,bhnk->bhmk', H, action)
            return this_obs, this_action
        return this_obs
    
    def convert2abs(self, x_curr, R_curr, action, obs=None):
        H = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,4,4).repeat(x_curr.shape[0], 1, 1)
        if self.relative_position:
            H[:,:3,3] = x_curr
        if self.relative_rotation:
            H[:,:3,:3] = R_curr
        this_action = torch.einsum('bmn,bhnk->bhmk', H, action)
        if obs is not None:
            this_obs = dict_apply(obs, lambda x: x.clone())
            this_obs['keypoint_pcd'] = torch.einsum('bmn,bhkn->bhkm', H, self.toHomogeneous(this_obs['keypoint_pcd']))[..., :3]
            this_obs['agent_pose'] = torch.einsum('bmn,bhnk->bhmk', H, this_obs['agent_pose'])
            return this_action, this_obs
        return this_action
    

    # ========= inference  ============    
    def get_random_pose(self, batch_size):
        R = torch.tensor(Rotation.random(batch_size).as_matrix()).to(self.device)
        t = torch.randn(batch_size, 3).to(self.device)
        H = torch.eye(4)[None,...].repeat(batch_size,1,1).to(self.device)
        H[:, :3, :3] = R
        H[:, :3, -1] = t
        return H

    def sample(self, global_cond, batch_size=64, To=1, T=100, get_traj=False):

        with torch.no_grad():
            self.model.set_context(global_cond)
            steps = T
            t = torch.linspace(0, 1., steps=steps)
            dt = 1/steps

            # Euler method
            # sample H_0 first
            H0 = self.get_random_pose(batch_size=batch_size*To)

            if get_traj:
                trj = H0[:,None,...].repeat(1,steps,1,1)
            Ht = H0
            for s in range(0, steps):
                ut = self.velocity_scale*self.model(Ht.view(batch_size,To,4,4), t[s]*torch.ones_like(Ht[:,0,0]))
                utdt = ut*dt

                utdt = utdt.reshape(batch_size*To, 6)

                ## rotation update ##
                rot_xt = Ht[:, :3, :3].cpu().numpy()
                rot_utdt = utdt[:, :3].cpu().numpy()
                rot_xt_v = self.vec_manifold.rotation_vector_from_matrix(rot_xt)
                rot_xt_v2 = self.vec_manifold.compose(rot_xt_v, rot_utdt)
                rot_xt_2 = self.vec_manifold.matrix_from_rotation_vector(rot_xt_v2)
                rot_xt_2 = torch.from_numpy(rot_xt_2).to(device=self.device, dtype=self.dtype)

                ## translation update ##
                trans_xt = Ht[:, :3, -1]
                trans_utdt = utdt[:, 3:]
                #Place the translation back in world frame
                #trans_utdt = torch.einsum('bij,bj->bi', rot_xt.transpose(-1,-2), trans_utdt)
                trans_xt2 = trans_xt + trans_utdt

                Ht[:, :3, :3] = rot_xt_2
                Ht[:, :3, -1] = trans_xt2
                if get_traj:
                    trj[:,s,...] = Ht
        
        Ht = Ht.reshape(batch_size, To, 4, 4)

        if get_traj:
            trj = trj.reshape(batch_size, steps, To, 4, 4)
            return Ht, trj
        else:
            return Ht

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input

        x_curr = obs_dict['agent_pose'][:,-1,:3,3].clone()
        R_curr = obs_dict['agent_pose'][:,-1,:3,:3].clone()
        if self.relative_position or self.relative_rotation:
            obs_dict = self.convert2rel(x_curr, R_curr, obs_dict)
        nobs = self.normalize_obs(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        T = self.horizon

        # build input
        device = self.device
        dtype = self.dtype
        nobs = dict_apply(nobs, lambda x: x.type(dtype).to(device))

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...]) # We need to keep the observation shape
        # (B, D)
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features

        # run sampling
        trajectory_pred = self.sample(global_cond, B, T)

        # regress gripper open
        gripper_state_ignore_collision_prediction = self.gripper_state_ignore_collision_predictor(nobs_features).reshape(B, T, 2)
        gripper_state_ignore_collision_prediction = torch.sigmoid(gripper_state_ignore_collision_prediction)
        gripper_state, ignore_collision = (gripper_state_ignore_collision_prediction > 0.5).float().chunk(2, dim=-1)
        
        # unnormalize prediction
        trajectory_pred = self.unnormalize_action(trajectory_pred)

        if self.relative_position or self.relative_rotation:
            trajectory_pred = self.convert2abs(x_curr, R_curr, trajectory_pred)
        
        action = action_from_trajectory_gripper_open_ignore_collision(trajectory_pred, gripper_state, ignore_collision)
        result = {
            'action': action,
            'trajectory': trajectory_pred,
            'open_gripper': gripper_state_ignore_collision_prediction[:,:,0],
            'ignore_collision': gripper_state_ignore_collision_prediction[:,:,1]
        }
        return result
    
    # ========= training  ============
    def augment_obs(self, obs):
        B = obs['agent_pose'].shape[0]
        R_delta = SO3().exp_map(torch.normal(mean=0, std=self.noise_aug_std, size=(B, 3))).to_matrix().to(self.device)
        x_delta = torch.normal(mean=0, std=self.noise_aug_std, size=(B, 3), device=self.device)
        H_delta = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,4,4).repeat(B, 1, 1)
        H_delta[:,:3,3] = x_delta
        H_delta[:,:3,:3] = R_delta
        obs["agent_pose"] = torch.einsum('bmn,bhnk->bhmk', H_delta, obs["agent_pose"])
        return obs
    
    def compute_loss(self, batch):
        # normalize input
        obs = batch['obs']
        actions = batch['action'].float()
        
        trajectory = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,1,4,4).repeat(actions.shape[0], self.horizon, 1, 1)
        trajectory[:,:,:3,3] = actions[:,:,:3]
        trajectory[:,:,:3,:3] = torch.from_numpy(Rotation.from_quat(actions[:,:,3:7].flatten(0,1).cpu()).as_matrix().reshape(actions.shape[0], self.horizon, 3, 3)).to(self.device, self.dtype)
        
        x_curr = obs['agent_pose'][:,-1,:3,3].clone()
        R_curr = obs['agent_pose'][:,-1,:3,:3].clone()

        if self.relative_position or self.relative_rotation:
            obs, trajectory = self.convert2rel(x_curr, R_curr, obs, trajectory)

        nobs = self.normalize_obs(obs)
        trajectory = self.normalize_trajectory(trajectory)

        # augment obs
        if self.data_augmentation:
            nobs = self.augment_obs(nobs)

        
        B = trajectory.shape[0]
        T = self.horizon
        To = self.n_obs_steps

        assert (T == trajectory.shape[1])

        # build input
        device = self.device
        dtype = self.dtype
        nobs = dict_apply(nobs, lambda x: x.type(dtype).to(device))
        trajectory = trajectory.type(dtype).to(device)

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...])
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features

        x = trajectory[:,:,:3,3]
        R = trajectory[:,:,:3,:3]

        H1 = trajectory.flatten(0,1)
        H0 = self.get_random_pose(B*To)
        t = torch.rand(H0.shape[0]).type_as(H0).to(H0.device)

        ## Sample X at time t through the Geodesic from x0 -> x1
        def sample_xt(x0, x1, t):
            """
            Function which compute the sample xt along the geodesic from x0 to x1 on SE(3).
            """
            ## Point to translation and rotation ##
            t0, R0 = x0[:, :3, -1], x0[:, :3, :3]
            t1, R1 = x1[:, :3, -1], x1[:, :3, :3]

            ## Get rot_t ##
            rot_x0 = self.vec_manifold.rotation_vector_from_matrix(R0.cpu().numpy())
            rot_x1 = self.vec_manifold.rotation_vector_from_matrix(R1.cpu().numpy())
            log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
            rot_xt = self.vec_manifold.exp_not_from_identity(t.cpu().reshape(-1, 1) * log_x1, rot_x0)
            rot_xt = self.vec_manifold.matrix_from_rotation_vector(rot_xt)
            rot_xt = torch.from_numpy(rot_xt).to(device)

            ## Get trans_t ##
            trans_xt = t0*(1. - t[:,None]) + t1*t[:,None]

            xt = torch.eye(4)[None,...].repeat(rot_xt.shape[0], 1, 1).to(device)
            xt[:, :3, :3] = rot_xt
            xt[:, :3, -1] = trans_xt
            return xt

        Ht = sample_xt(H0.double(), H1.double(), t)

        ## Compute velocity target at xt at time t through the geodesic x0 -> x1
        def compute_conditional_vel(x0, x1, xt, t):

            def invert_se3(matrix):
                """
                Invert a homogeneous transformation matrix.

                :param matrix: A 4x4 numpy array representing a homogeneous transformation matrix.
                :return: The inverted transformation matrix.
                """

                # Extract rotation (R) and translation (t) from the matrix
                R = matrix[..., :3, :3]
                t = matrix[..., :3, 3]

                # Invert the rotation (R^T) and translation (-R^T * t)
                R_inv = torch.transpose(R, -1, -2)
                t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)

                # Construct the inverted matrix
                inverted_matrix = torch.clone(matrix)
                inverted_matrix[..., :3, :3] = R_inv
                inverted_matrix[..., :3, 3] = t_inv

                return inverted_matrix


            xt_inv = invert_se3(xt)

            # xt1 = torch.einsum('...ij,...jk->...ik', xt_inv, x1)
            ## Point to translation and rotation ##
            # trans_xt1, rot_xt1 = xt1[:, :3, -1], xt1[:, :3, :3]

            trans_xt, rot_xt = xt[:, :3, -1], xt[:, :3, :3]
            trans_x1, rot_x1 = x1[:, :3, -1], x1[:, :3, :3]

            ## Compute Velocity in rot ##
            delta_r = torch.transpose(rot_xt, -1, -2)@rot_x1
            rot_ut = torch.from_numpy(self.vec_manifold.rotation_vector_from_matrix(delta_r.cpu().numpy())).to(self.device) / torch.clip((1 - t[:, None]), 0.01, 1.)

            ## Compute Velocity in trans ##
            #trans_ut = -trans_xt1/ torch.clip((1 - t[:, None]), 0.01, 1.)
            trans_ut = (trans_x1 - trans_xt)/ torch.clip((1 - t[:, None]), 0.01, 1.)
            #trans_ut = 0.*trans_ut

            return torch.cat((rot_ut, trans_ut), dim=1).detach()

        ut = compute_conditional_vel(H0.double(), H1.double(), Ht.double(), t)
        ut = ut.reshape(B, -1)

        # regress gripper open
        pred = self.gripper_state_ignore_collision_predictor(nobs_features).reshape(B, T, 2)
        regression_loss = F.binary_cross_entropy_with_logits(pred, actions[:,:,7:9])

        # Predict the noise residual
        Ht.requires_grad = True
        t.requires_grad = True
        Ht = Ht.reshape(B, To, 4, 4)
        self.model.set_context(nobs_features)
        vt = self.model(Ht, t)

        loss = torch.mean((ut-vt)**2)
        loss += regression_loss * 0.1
        return loss

import hydra
from omegaconf import OmegaConf
import pathlib

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config')),
    config_name='train_flow_matching_unet_lowdim_workspace.yaml'
)
def test(cfg: OmegaConf):
    from copy import deepcopy
    OmegaConf.resolve(cfg)
    policy : FlowMatchingUnetLowDimPolicy = hydra.utils.instantiate(cfg.policy)
    policy = policy.to(cfg.training.device)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = dict_apply(dataset[0], lambda x: x.unsqueeze(0).float().cuda())
    action= batch['action']    
    trajectory = torch.eye(4).reshape(1,1,4,4).repeat(action.shape[0], policy.horizon, 1, 1).cuda()
    trajectory[:,:,:3,3] = action[:,:,:3]
    trajectory[:,:,:3,:3] = torch.from_numpy(Rotation.from_quat(action[:,:,3:7].flatten(0,1).cpu()).as_matrix().reshape(action.shape[0], policy.horizon, 3, 3)).cuda()
    obs = batch['obs']
    x_curr = obs['agent_pose'][:,-1,:3,3]
    R_curr = obs['agent_pose'][:,-1,:3,:3]
    robs, rtrajectory = policy.convert2rel(x_curr, R_curr, deepcopy(obs),deepcopy(trajectory))
    atrajectory, aobs = policy.convert2abs(x_curr, R_curr, deepcopy(rtrajectory), deepcopy(robs))

    assert(torch.allclose(aobs['agent_pose'], obs['agent_pose'])), "agent_pose not equal with max diff: %f" % (aobs['agent_pose'] - obs['agent_pose']).abs().max()
    assert(torch.allclose(atrajectory, trajectory)), "action not equal with max diff: %f" % (atrajectory - trajectory).abs().max()
    assert (torch.allclose(aobs['keypoint_pcd'], obs['keypoint_pcd'])), "keypoint_pcd not equal with max diff: %f" % (aobs['keypoint_pcd'] - obs['keypoint_pcd']).abs().max()
    policy.compute_loss(batch)
    policy.predict_action(obs)
    print("Test passed")

if __name__ == "__main__":
    test()