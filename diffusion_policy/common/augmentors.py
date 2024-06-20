import torch
from diffusion_policy.model.common.so3_util import normal_so3

class AugmentGripperHistory():
    def __init__(self, rot_scale, pos_scale):
        self.rot_scale = rot_scale
        self.pos_scale = pos_scale

    def augment(self, batch):
        obs = batch['obs']
        b, n_obs = obs['robot0_eef_rot'].size(0), obs['robot0_eef_rot'].size(1)
        if self.rot_scale != 0:
            gripper_rot = obs['robot0_eef_rot']
            delta_rot = normal_so3(b * n_obs , scale=self.rot_scale)
            delta_rot = delta_rot.reshape(b, n_obs, 3, 3)
            delta_rot = delta_rot.to(gripper_rot.device)
            gripper_rot = torch.matmul(delta_rot, gripper_rot)
        if self.pos_scale != 0:
            gripper_pos = obs['robot0_eef_pos']
            delta_pos = torch.randn(b, n_obs, 3) * self.pos_scale
            delta_pos = delta_pos.to(gripper_pos.device)
            gripper_pos = gripper_pos + delta_pos
        return batch
