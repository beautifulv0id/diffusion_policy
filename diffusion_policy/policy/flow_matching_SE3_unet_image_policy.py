from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.rlbench_util import action_from_trajectory_gripper_open_ignore_collision
from diffusion_policy.common.pytorch_util import dict_apply
from typing import Union, Dict, Optional
from scipy.spatial.transform import Rotation 
from pytorch3d.transforms import quaternion_to_matrix
from diffusion_policy.common.rotation_utils import SO3_exp_map, sample_random_se3
from diffusion_policy.common.flow_matching_utils import sample_xt_SE3, compute_conditional_vel_SE3
from diffusion_policy.model.vision.transformer_feature_pointcloud_encoder import TransformerFeaturePointCloudEncoder

def fill_mask(mask : torch.Tensor):
    mask_shape = mask.shape
    mask = mask.flatten(1)
    batch_size, n = mask.shape
    false_idx = (~mask).nonzero(as_tuple=True)
    true_row = mask.sum(dim=-1).cpu().int().numpy()
    false_row = n - true_row
    cum_sum = np.cumsum([0] + false_row.tolist()).astype(int)
    max_true = np.max(true_row)
    n_pad = max_true - true_row
    pad_idx = ([], [])
    for i in range(batch_size):
        from_idx = cum_sum[i]
        to_idx = cum_sum[i+1]
        idx = np.random.choice(np.arange(from_idx, to_idx), n_pad[i], replace=False)
        pad_idx[0].extend(false_idx[0][idx])
        pad_idx[1].extend(false_idx[1][idx])
    padded_mask = mask.clone()
    padded_mask[pad_idx] = 1
    return padded_mask.reshape(mask_shape)


def mask_and_fill_remaining(vals : torch.Tensor, mask : torch.Tensor):
    padded_mask = fill_mask(mask)
    padded_vals = vals[padded_mask].reshape(vals.shape[0], -1)
    return padded_vals

def crop_to_bounds(visual_features, pcds, bounds=None):
    if bounds is None:
        bounds = torch.tensor([[-1, -1, -1], [1, 1, 1]], device=pcds.device, dtype=pcds.dtype) 
    batch_size, channels = visual_features.shape[0], visual_features.shape[-1]
    pos_min = bounds[0].float().to(pcds.device)
    pos_max = bounds[1].float().to(pcds.device)

    mask = (pcds[..., :3] > pos_min) & (pcds[..., :3] < pos_max)
    mask = mask.all(dim=-1)
    mask = fill_mask(mask)
    if not mask.any():
        return visual_features, pcds
    visual_features = visual_features[mask].reshape(batch_size, -1, channels)
    pcds = pcds[mask].reshape(batch_size, -1, 3)

    return visual_features, pcds

class FlowMatchingSE3UnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            observation_encoder: Union[nn.Module, TransformerFeaturePointCloudEncoder],
            model: nn.Module,
            horizon : int, 
            n_action_steps : int, 
            n_obs_steps : int,
            num_inference_steps=10,
            delta_t=0.01,
            gripper_loc_bounds=None,
            relative_position=True,
            relative_rotation=False,
            noise_aug_std=0.1,
            velocity_scale=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 2
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'pcd': [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'low_dim':
                obs_config['low_dim'].append(key)
            elif type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'pcd':
                obs_config['pcd'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # create diffusion model
        obs_feature_dim = observation_encoder.output_shape()[0]

        self.encoder = observation_encoder
        self.model = model
        self.ignore_collision_predictor = nn.Sequential(
            nn.Linear(obs_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
        )
        self.open_gripper_predictor = nn.Sequential(
            nn.Linear(obs_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, horizon),
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
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

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.encoder.parameters()))

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
    
    def convert_action(self, action):
        trajectory = torch.eye(4, device=self.device, dtype=self.dtype).reshape(1,1,4,4).repeat(action.shape[0], self.horizon, 1, 1)
        trajectory[:,:,:3,3] = action[:,:,:3]
        trajectory[:,:,:3,:3] = quaternion_to_matrix(torch.cat([action[:,:,6:7], action[:,:,3:6]], dim=-1))
        return trajectory
        
    def normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict['agent_pose'][:,:,:3,3] = self.normalize_pos(obs_dict['agent_pose'][:,:,:3,3])
        obs_dict['pcd'][:,:,:3] = self.normalize_pos(obs_dict['pcd'])
        return obs_dict
        
    def toHomogeneous(self, x):
        return torch.cat([x, torch.ones_like(x[...,0:1])], dim=-1)
    
    def convert2rel_pos(self, gripper, pcd, trajectory=None):
        center = gripper[:,-1,:3,3]
        gripper = gripper.clone()
        pcd = pcd.clone()
        pcd = pcd - center.view(-1,1,3)
        gripper[:,:,:3,3] = gripper[:,:,:3,3] - center.view(-1,1,3)
        if trajectory is None:
            return gripper, pcd
        else:
            trajectory = trajectory.clone()
            trajectory[:,:,:3,3] = trajectory[:,:,:3,3] - center.view(-1,1,3)
            return gripper, pcd, trajectory
    
    def convert2abs_pos(self, gripper, trajectory, pcd=None):
        center = gripper[:,-1,:3,3]
        trajectory = trajectory.clone()
        trajectory[:,:,:3,3] = trajectory[:,:,:3,3] + center.view(-1,1,3)
        if pcd is None:
            return trajectory
        else:
            pcd = pcd.clone()
            pcd = pcd + center.view(-1,1,3)
            return trajectory, pcd
    
    def convert2rel_rot(self, gripper, pcd, trajectory=None):
        R = gripper[:,-1,:3,:3].transpose(-1,-2)
        gripper = gripper.clone()
        pcd = torch.einsum('bmn,bln->blm', R, pcd)
        gripper[:,:,:3,:] = torch.einsum('bmn,bhnk->bhmk', R, gripper[:,:,:3,:])
        if trajectory is None:
            return gripper, pcd
        else:
            trajectory = trajectory.clone()
            trajectory[:,:,:3,:] = torch.einsum('bmn,bhnk->bhmk', R, trajectory[:,:,:3,:])
            return gripper, pcd, trajectory
    
    def convert2abs_rot(self, gripper, trajectory, pcd=None):
        R = gripper[:,-1,:3,:3]
        trajectory = trajectory.clone()
        trajectory[:,:,:3,:] = torch.einsum('bmn,bhnk->bhmk', R, trajectory[:,:,:3,:])
        if pcd is None:
            return trajectory
        else:
            pcd = torch.einsum('bmn,bln->blm', R, pcd)
            return trajectory, pcd
    
    def get_random_pose(self, batch_size):
        lR = torch.randn((batch_size, 3), device=self.device, dtype=self.dtype)
        R = SO3_exp_map(lR)
        t = torch.randn((batch_size, 3), device=self.device, dtype=self.dtype)
        H = torch.eye(4, device=self.device, dtype=self.dtype)[None,...].repeat(batch_size,1,1)
        H[:, :3, :3] = R
        H[:, :3, -1] = t
        return H

    # ========= inference  ============    
    def sample(self, global_cond, batch_size=64, horizon=1, T=100, get_traj=False):

        with torch.no_grad():
            steps = T
            t = torch.linspace(0, 1., steps=steps)
            dt = 1/(steps-1)

            # Euler method
            # sample H_0 first
            H0 = self.get_random_pose(batch_size=batch_size*horizon)

            if get_traj:
                trj = H0[:,None,...].repeat(1,steps,1,1)
            Ht = H0
            for s in range(0, steps):
                ut = self.velocity_scale*self.model(
                                                Ht.view(batch_size,horizon,4,4), 
                                                t[s]*torch.ones_like(Ht[:,0,0]), 
                                                global_cond)
                utdt = ut*dt
                utdt = utdt.reshape(batch_size*horizon, 6)

                ## rotation update ##
                R_w_t = Ht[:, :3, :3]
                R_w_tp1 = SO3_exp_map(utdt[:, :3])
                R_w_tp1 = torch.matmul(R_w_t, R_w_tp1)           

                ## translation update ##
                x_t = Ht[:, :3, -1]
                x_utdt = utdt[:, 3:]
                x_tp1 = x_t + x_utdt

                Ht[:, :3, :3] = R_w_tp1
                Ht[:, :3, -1] = x_tp1
                if get_traj:
                    trj[:,s,...] = Ht
        
        Ht = Ht.reshape(batch_size, horizon, 4, 4)

        if get_traj:
            trj = trj.reshape(batch_size, steps, horizon, 4, 4)
            return Ht, trj
        else:
            return Ht

    def predict_action(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        obs = dict_apply(obs, lambda x: x.clone().type(self.dtype).to(self.device))

        gripper = obs['agent_pose']
        pcd = obs['pcd']
        rgb = obs['rgb']
        encoder = self.encoder

        # encode observation
        visual_features = encoder.encode_rgbs(rgb)
        pcd = encoder.interpolate_pcds(pcd, visual_features.shape[-2:])
        visual_features = einops.rearrange(visual_features, "bt ncam c h w -> bt (ncam h w) c")
        pcd = einops.rearrange(pcd, "bt ncam c h w -> bt (ncam h w) c")

        # crop to workspace
        visual_features, pcd = crop_to_bounds(visual_features, pcd, self.gripper_loc_bounds)

        # normalize input
        gripper[:,:,:3,3] = self.normalize_pos(gripper[:,:,:3,3])
        gripper_abs = gripper.clone()
        pcd = self.normalize_pos(pcd)

        if self.relative_position:
            gripper, pcd = self.convert2rel_pos(gripper_abs, pcd)
        if self.relative_rotation:
            gripper, pcd = self.convert2rel_rot(gripper_abs, pcd)

        value = next(iter(obs.values()))
        B = value.shape[0]
        T = self.horizon

        # encode gripper
        gripper_features, _ = self.encoder._encode_gripper(gripper, visual_features, pcd)
        gripper_features = self.encoder._encode_current_gripper(gripper_features)

        # run sampling
        trajectory_pred = self.sample(gripper_features, B, T, self.num_inference_steps)

        # regress gripper open
        gripper_state_prediction = torch.sigmoid(self.open_gripper_predictor(gripper_features))
        gripper_state = (gripper_state_prediction > 0.5).float()
        ignore_collision_prediction = torch.sigmoid(self.ignore_collision_predictor(gripper_features))
        ignore_collision = (ignore_collision_prediction > 0.5).float()
        
        if self.relative_rotation:
            trajectory_pred = self.convert2abs_rot(gripper_abs, trajectory_pred)
        if self.relative_position:
            trajectory_pred = self.convert2abs_pos(gripper_abs, trajectory_pred)

        # unnormalize prediction
        trajectory_pred[:,:,:3,3] = self.unnormalize_pos(trajectory_pred[:,:,:3,3])
        
        action = action_from_trajectory_gripper_open_ignore_collision(trajectory_pred, gripper_state, ignore_collision)
        result = {
            'action': action,
            'trajectory': trajectory_pred,
            'open_gripper': gripper_state_prediction,
            'ignore_collision': ignore_collision_prediction
        }
        return result
    
    # ========= training  ============
    def apply_noise(self, obs):
        agent_pose = obs['agent_pose']
        B = agent_pose.shape[0]
        R = SO3_exp_map(torch.normal(mean=0, std=self.noise_aug_std, size=(B, 3), device=self.device))
        x = torch.normal(mean=0, std=self.noise_aug_std, size=(B, 1, 3), device=self.device)
        agent_pose[:,-1,:3,:3] = torch.einsum('bmn,bnk->bhmk', R, agent_pose[:,-1,:3,:3])
        agent_pose[:,:3,3] = agent_pose[:,:3,3] + x
        obs['agent_pose'] = agent_pose
        return obs
    
    def compute_loss(self, batch):
        obs = dict_apply(batch['obs'], lambda x: x.clone().type(self.dtype).to(self.device))
        actions = batch['action'].clone().type(self.dtype).to(self.device)

        if self.data_augmentation:
            obs = self.apply_noise(obs)

        # prepare input
        pcd = obs['pcd']
        rgb = obs['rgb']
        gripper = obs['agent_pose']
        encoder = self.encoder

        # encode observation
        visual_features = encoder.encode_rgbs(rgb)
        pcd = encoder.interpolate_pcds(pcd, visual_features.shape[-2:])
        visual_features = einops.rearrange(visual_features, "bt ncam c h w -> bt (ncam h w) c")
        pcd = einops.rearrange(pcd, "bt ncam c h w -> bt (ncam h w) c")

        # crop to workspace
        visual_features, pcd = crop_to_bounds(visual_features, pcd, self.gripper_loc_bounds)

        # normalize input
        gripper[:,:,:3,3] = self.normalize_pos(gripper[:,:,:3,3])
        pcd = self.normalize_pos(pcd)
        actions[:,:,:3] = self.normalize_pos(actions[:,:,:3])

        # prepare targets
        trajectory = self.convert_action(actions) # pos+quat -> SE3
        open_gripper = actions[:,:,7]
        ignore_collision = actions[:,:,8]

        B = trajectory.shape[0]
        T = self.horizon
        assert (T == trajectory.shape[1])

        # if True:
        #     H = sample_random_se3(B, 0.01, 0.01, self.device, self.dtype)
        #     pcd = torch.einsum('bmn,bln->blm', H, self.toHomogeneous(pcd))[...,:3]
        #     gripper = torch.einsum('bmn,bhnk->bhmk', H, gripper)
        #     trajectory = torch.einsum('bmn,bhnk->bhmk', H, trajectory)
        
        # convert to relative
        if self.relative_position:
            gripper, pcd, trajectory = self.convert2rel_pos(gripper, pcd, trajectory)
        if self.relative_rotation:
            gripper, pcd, trajectory = self.convert2rel_rot(gripper, pcd, trajectory)
        
        # encode gripper
        gripper_features, _ = self.encoder._encode_gripper(gripper, visual_features, pcd)
        gripper_features = self.encoder._encode_current_gripper(gripper_features)

        # compute conditional velocity
        H1 = trajectory.flatten(0,1)
        H0 = self.get_random_pose(B*T)
        t = torch.rand(B, device=self.device, dtype=self.dtype)
        Ht = sample_xt_SE3(H0, H1, t)
        ut = compute_conditional_vel_SE3(H1, Ht, t)
        ut = ut.reshape(B, -1)

        # Predict the noise residual
        # Ht.requires_grad = True
        # t.requires_grad = True
        Ht = Ht.reshape(B, T, 4, 4)
        vt = self.model(Ht, t, gripper_features)

        # regress gripper open
        gripper_open_pred = self.open_gripper_predictor(gripper_features)
        ignore_collision_pred = self.ignore_collision_predictor(gripper_features)

        open_gripper_loss = F.binary_cross_entropy_with_logits(gripper_open_pred, open_gripper)
        ignore_collision_loss = F.binary_cross_entropy_with_logits(ignore_collision_pred, ignore_collision)

        loss = torch.mean((ut-vt)**2)
        loss += 0.1 * (open_gripper_loss + ignore_collision_loss)
        return loss

import hydra
from omegaconf import OmegaConf
import pathlib

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config')),
    config_name='train_flow_matching_unet_image_workspace.yaml'
)
def test(cfg: OmegaConf):
    from copy import deepcopy
    OmegaConf.resolve(cfg)
    policy : FlowMatchingSE3UnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    policy = policy.to(cfg.training.device)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = dict_apply(dataset[0], lambda x: x.unsqueeze(0).float().cuda())
    action= batch['action']    
    trajectory = policy.convert_action(action)
    obs = batch['obs']
    pcd = einops.rearrange(obs['pcd'], 'b ncam c h w -> b (ncam h w) c')
    gripper = obs['agent_pose']
    rgripper, rpcd, rtrajectory = policy.convert2rel_pos(gripper, pcd, trajectory)
    atrajectory, apcd = policy.convert2abs_pos(gripper, rtrajectory, rpcd)
    assert((apcd - pcd).abs().max() < 1e-6), "pcd not equal with max diff: %f" % (apcd - pcd).abs().max()
    assert((atrajectory - trajectory).abs().max() < 1e-6), "trajectory not equal with max diff: %f" % (atrajectory - trajectory).abs().max()

    rgripper, rpcd, rtrajectory = policy.convert2rel_rot(gripper, pcd, trajectory)
    atrajectory, apcd = policy.convert2abs_rot(gripper, rtrajectory, rpcd)
    assert((apcd - pcd).abs().max() < 1e-6), "pcd not equal with max diff: %f" % (apcd - pcd).abs().max()
    assert((atrajectory - trajectory).abs().max() < 1e-6), "trajectory not equal with max diff: %f" % (atrajectory - trajectory).abs().max()

    rgripper, rpcd, rtrajectory = policy.convert2rel_pos(gripper, pcd, trajectory)
    rgripper, rpcd, rtrajectory = policy.convert2rel_rot(gripper, rpcd, rtrajectory)
    atrajectory, apcd = policy.convert2abs_rot(gripper, rtrajectory, rpcd)
    atrajectory, apcd = policy.convert2abs_pos(gripper, atrajectory, apcd)
    assert((apcd - pcd).abs().max() < 1e-6), "pcd not equal with max diff: %f" % (apcd - pcd).abs().max()
    assert((atrajectory - trajectory).abs().max() < 1e-6), "trajectory not equal with max diff: %f" % (atrajectory - trajectory).abs().max()


    policy.compute_loss(batch)
    policy.predict_action(obs)
    print("Test passed")

if __name__ == "__main__":
    test()