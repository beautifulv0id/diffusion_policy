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
from diffusion_policy.model.flow_matching.naive_film_se3 import NaiveFilmSE3FlowMatchingModel
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.transformer_lowdim_obs_relative_encoder import TransformerLowdimObsRelativeEncoder
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.common.common_utils import action_from_trajectory_gripper_open_ignore_collision
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.se3_diffusion_util import marginal_prob_std, sample_from_se3_gaussian, se3_score_normal, step
from theseus.geometry.so3 import SO3
from scipy.spatial.transform import Rotation 
from pytorch3d.transforms import quaternion_to_matrix

def rotation_vector_from_matrix(R):
    """
    Convert a batch of rotation matrices to rotation vectors.
    """
    R_ = SO3()
    R_.update(R)
    return R_.log_map()

def matrix_from_rotation_vector(rot):
    """
    Convert a batch of rotation vectors to rotation matrices.
    """
    return SO3().exp_map(rot).to_matrix()
class FlowMatchingUnetLowDimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            shape_meta: dict,
            observation_encoder: TransformerLowdimObsRelativeEncoder,
            # action_decoder: nn.Module,
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
        obs_feature_dim = observation_encoder.output_shape()[0]
        input_dim = 16 * horizon

        action_decoder = NaiveFilmSE3FlowMatchingModel(
            in_channels=input_dim,
            out_channels=6,
            embed_dim=obs_feature_dim,
            cond_dim=obs_feature_dim
        )

        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
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
        if noise_aug_std > 0:
            self.noise_aug_std = noise_aug_std
            self.data_augmentation = True
        else:
            self.data_augmentation = False

        print("Diffusion params: %e" % sum(p.numel() for p in self.action_decoder.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.observation_encoder.parameters()))

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
        lR = torch.randn((batch_size, 3), device=self.device, dtype=self.dtype)
        R = SO3.exp_map(lR).to_matrix()
        t = torch.randn((batch_size, 3), device=self.device, dtype=self.dtype)
        H = torch.eye(4, device=self.device, dtype=self.dtype)[None,...].repeat(batch_size,1,1)
        H[:, :3, :3] = R
        H[:, :3, -1] = t
        return H

    def sample(self, global_cond, batch_size=64, To=1, T=100, get_traj=False):

        with torch.no_grad():
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
                ut = self.velocity_scale*self.action_decoder(Ht.view(batch_size,To,4,4), t[s]*torch.ones_like(Ht[:,0,0]), global_cond)
                utdt = ut*dt

                utdt = utdt.reshape(batch_size*To, 6)

                ## rotation update ##
                R_w_t = Ht[:, :3, :3]
                R_w_tp1 = SO3.exp_map(utdt[:, :3]).to_matrix()
                R_w_tp1 = torch.matmul(R_w_t, R_w_tp1)           

                ## translation update ##
                x_t = Ht[:, :3, -1]
                x_utdt = utdt[:, 3:]
                x_tp1 = x_t + x_utdt

                Ht[:, :3, :3] = R_w_tp1
                Ht[:, :3, -1] = x_tp1
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
        nobs_features = self.observation_encoder(this_nobs)
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
        R = SO3().exp_map(torch.normal(mean=0, std=self.noise_aug_std, size=(B, 3), device=self.device)).to_matrix()
        x = torch.normal(mean=0, std=self.noise_aug_std, size=(B, 3), device=self.device)
        H_delta = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,4,4).repeat(B, 1, 1)
        H_delta[:,:3,3] = x
        H_delta[:,:3,:3] = R
        obs["agent_pose"] = torch.einsum('bmn,bhnk->bhmk', H_delta, obs["agent_pose"])
        return obs
    
    def compute_loss(self, batch):
        # normalize input
        obs = batch['obs']
        actions = batch['action'].float()
        
        trajectory = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,1,4,4).repeat(actions.shape[0], self.horizon, 1, 1)
        trajectory[:,:,:3,3] = actions[:,:,:3]
        trajectory[:,:,:3,:3] = quaternion_to_matrix(torch.cat([actions[:,:,6:7], actions[:,:,3:6]], dim=-1))

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
        nobs_features = self.observation_encoder(this_nobs)
        global_cond = nobs_features

        H1 = trajectory.flatten(0,1)
        H0 = self.get_random_pose(B*To)
        t = torch.rand(H0.shape[0], device=device, dtype=dtype)

        ## Sample X at time t through the Geodesic from x0 -> x1
        def sample_xt(x0, x1, t):
            """
            Function which compute the sample xt along the geodesic from x0 to x1 on SE(3).
            """
            ## Point to translation and rotation ##
            t_0, R_w_0 = x0[:, :3, -1], x0[:, :3, :3]
            t_1, R_w_1 = x1[:, :3, -1], x1[:, :3, :3]

            ## Get rot_t ##
            R_0_w = torch.transpose(R_w_0, -1, -2)
            R_0_1 = torch.matmul(R_0_w,R_w_1)
            lR_0_1 = rotation_vector_from_matrix(R_0_1)
            R_0_t = matrix_from_rotation_vector(lR_0_1*t[:,None])
            R_w_t = torch.matmul(R_w_0, R_0_t)

            ## Get trans_t ##
            x_t = t_0*(1. - t[:,None]) + t_1*t[:,None]

            Ht = torch.eye(4, device=device)[None,...].repeat(R_w_t.shape[0], 1, 1)
            Ht[:, :3, :3] = R_w_t
            Ht[:, :3, -1] = x_t
            return Ht

        Ht = sample_xt(H0, H1, t)

        ## Compute velocity target at xt at time t through the geodesic x0 -> x1
        def compute_conditional_vel(x0, H_w_1, H_w_t, t):
            x_t, R_w_t = H_w_t[:, :3, -1], H_w_t[:, :3, :3]
            x_1, R_w_1 = H_w_1[:, :3, -1], H_w_1[:, :3, :3]

            ## Compute Velocity in rot ##
            R_t_w = torch.transpose(R_w_t, -1, -2)
            R_t_1 = torch.matmul(R_t_w, R_w_1)
            lR_t_ut = rotation_vector_from_matrix(R_t_1) / torch.clip((1 - t[:, None]), 0.01, 1.)

            ## Compute Velocity in trans ##
            x_ut = (x_1 - x_t)/ torch.clip((1 - t[:, None]), 0.01, 1.)

            return torch.cat((lR_t_ut, x_ut), dim=1).detach()

        ut = compute_conditional_vel(H0, H1, Ht, t)
        ut = ut.reshape(B, -1)

        # regress gripper open
        pred = self.gripper_state_ignore_collision_predictor(global_cond).reshape(B, T, 2)
        regression_loss = F.binary_cross_entropy_with_logits(pred, actions[:,:,7:9])

        # Predict the noise residual
        Ht.requires_grad = True
        t.requires_grad = True
        Ht = Ht.reshape(B, To, 4, 4)
        vt = self.action_decoder(Ht, t, global_cond)

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