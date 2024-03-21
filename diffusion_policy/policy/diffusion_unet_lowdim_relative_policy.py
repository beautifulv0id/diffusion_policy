from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.simple_network import NaiveConditionalSE3DiffusionModel
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

class DiffusionUnetLowDimRelativePolicy(BaseImagePolicy):
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
        input_dim = action_dim


        model = NaiveConditionalSE3DiffusionModel(
            in_channels=input_dim,
            out_channels=6,
            embed_dim=obs_feature_dim,
            cond_dim=obs_feature_dim
        )

        self.obs_encoder = obs_encoder
        self.model : NaiveConditionalSE3DiffusionModel = model
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
    def sample(self, B, T, global_cond):
        model = self.model

        # R0 = SO3.rand(B*T).to_matrix().to(global_cond.device)
        R0 = SO3().exp_map(torch.randn((B*T,3))).to_matrix().to(global_cond.device)
        x0 = torch.randn(B*T, 3).to(global_cond.device)
        
        K = self.num_inference_steps
        dt = self.dt
        for k in range(K):
            t = (K - k)/K + 10e-3
            t = torch.tensor(t, device=global_cond.device).repeat(B)
            v = model(x0.reshape(B,-1), R0.reshape(B,3,3), t, global_cond=global_cond)
            _s = v*dt
            x0, R0 = step(x0, R0, _s.reshape(B*T, 6))

        trajectory = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,1,4,4).repeat(B, T, 1, 1)
        trajectory[:,:,:3,3] = x0.reshape(B, T, 3)
        trajectory[:,:,:3,:3] = R0.reshape(B, T, 3, 3)
        return trajectory

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
        trajectory_pred = self.sample(B, T, global_cond=global_cond)

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
        # Sample noise that we'll add to the trajectory
        t = torch.rand(B, device=trajectory.device) + 10e-3
        std = marginal_prob_std(t, sigma=0.5)
        # Compute score
        with torch.enable_grad():
            noisy_x, noisy_R = sample_from_se3_gaussian(x.flatten(0,1), R.flatten(0,1), std.repeat_interleave(T, dim=0))
            v_tar = se3_score_normal(noisy_x, noisy_R, x_tar=x.flatten(0,1), R_tar=R.flatten(0,1), std=std.repeat_interleave(T, dim=0))

        # regress gripper open
        pred = self.gripper_state_ignore_collision_predictor(nobs_features).reshape(B, T, 2)
        regression_loss = F.binary_cross_entropy_with_logits(pred, actions[:,:,7:9])

        # TODO: generalize to T > 1
        v_tar = v_tar.view(B,-1)

        # Predict the noise residual
        v_pred = self.model(noisy_x.reshape(B,-1), noisy_R.reshape(B,-1,3,3), t, global_cond=global_cond)
        
        # TODO: check why
        loss = ((std.pow(2))*(v_pred - v_tar).pow(2).sum(-1)).mean()
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
    config_name='train_diffusion_unet_lowdim_relative_workspace.yaml'
)
def test(cfg: OmegaConf):
    from copy import deepcopy
    OmegaConf.resolve(cfg)
    policy : DiffusionUnetLowDimRelativePolicy = hydra.utils.instantiate(cfg.policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = dict_apply(dataset[0], lambda x: x.unsqueeze(0).float())
    action= batch['action']    
    trajectory = torch.eye(4).reshape(1,1,4,4).repeat(action.shape[0], policy.horizon, 1, 1)
    trajectory[:,:,:3,3] = action[:,:,:3]
    trajectory[:,:,:3,:3] = torch.from_numpy(Rotation.from_quat(action[:,:,3:7].flatten(0,1)).as_matrix().reshape(action.shape[0], policy.horizon, 3, 3)    )
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