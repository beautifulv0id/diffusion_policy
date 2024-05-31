import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.so3_util import log_map, exp_map, se3_inverse, apply_transform
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion
from diffusion_policy.common.se3_util import se3_from_rot_pos, rot_pos_from_se3

class SequenceAdaptor(ModuleAttrMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        adaptors = []
        for arg in args:
            adaptors.append(arg)
        self.adaptors = adaptors


    def adapt(self, data):
        for adaptor in self.adaptors:
            data = adaptor.adapt(data)
        return data

    def unadapt(self, data):
        for adaptor in self.adaptors[::-1]:
            data = adaptor.unadapt(data)
        return data

class Robomimic2Peract(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    def adapt(self, data):
        new_data = {}
        for k, v in data.items():
            if k != 'action' and k != 'obs':
                new_data[k] = v

        if 'obs' in data:
            new_obs = {}
            gripper_pos = data['obs']['robot0_eef_pos']
            gripper_rot = data['obs']['robot0_eef_rot']
            gripper_H = se3_from_rot_pos(gripper_rot, gripper_pos)
            new_obs['agent_pose'] = gripper_H

            keypoint_poses = torch.tensor([], device=gripper_pos.device, dtype=gripper_pos.dtype)
            kp_idx = 0
            while 'kp%d_pos' % kp_idx in data['obs']:
                kp_pos = data['obs']['kp%d_pos' % kp_idx]
                kp_rot = data['obs']['kp%d_rot' % kp_idx]
                kp_H = se3_from_rot_pos(kp_rot, kp_pos)
                keypoint_poses = torch.cat([keypoint_poses, kp_H], dim=1)
                kp_idx += 1
            new_obs['keypoint_poses'] = keypoint_poses

            if 'low_dim_state' in data['obs']:
                new_obs['low_dim_state'] = data['obs']['low_dim_state']
            if 'low_dim_pcd' in data['obs']:
                new_obs['low_dim_pcd'] = data['obs']['low_dim_pcd'][:,-1]
            if 'rgb' in data['obs']:
                new_obs['rgb'] = data['obs']['rgb'][:,-1]
            if 'pcd' in data['obs']:
                new_obs['pcd'] = data['obs']['pcd'][:,-1]
        
            new_data['obs'] = new_obs

        if 'action' in data:        
            action = data['action']

            ## Target Positions and Rotation ##
            act_p = action['act_p']
            act_r = action['act_r']
            act_gr = action['act_gr']
            act_ic = action['act_ic']

            ## Convert Rotation to Quaternion ##
            act_quat = standardize_quaternion(matrix_to_quaternion(act_r)) # TODO: might need to transpose
            new_action = torch.cat([act_p, act_quat, act_gr.unsqueeze(-1), act_ic.unsqueeze(-1)], dim=-1)

            new_data['action'] = new_action

        return new_data
    
    def unadapt(self, data):
        new_data = {}
        for k, v in data.items():
            if k != 'action' and k != 'obs':
                new_data[k] = v

        if 'action' in data:
            action = data['action']

            ## Target Positions and Rotation ##
            new_action = {}
            act_p = action[..., :3]
            act_quat = action[..., 3:7]
            act_r = standardize_quaternion(quaternion_to_matrix(act_quat))
            act_gr = action[..., 7]
            act_ic = action[..., 8]
            new_action['act_p'] = act_p
            new_action['act_r'] = act_r
            new_action['act_gr'] = act_gr
            new_action['act_ic'] = act_ic

            new_data['action'] = new_action

        if 'obs' in data:
            new_obs = {}
            gripper_H = data['obs']['agent_pose']
            gripper_rot, gripper_pos = rot_pos_from_se3(gripper_H)
            new_obs['robot0_eef_pos'] = gripper_pos
            new_obs['robot0_eef_rot'] = gripper_rot
            kp_idx = 0
            keypoint_pose = data['obs']['keypoint_poses']
            for i in range(keypoint_pose.shape[1]):
                kp_rot, kp_pos = rot_pos_from_se3(keypoint_pose[:, i, :, :])
                new_obs['kp%d_pos' % kp_idx] = kp_pos.unsqueeze(1)
                new_obs['kp%d_rot' % kp_idx] = kp_rot.unsqueeze(1)
                kp_idx += 1
            if 'low_dim_state' in data['obs']:
                new_obs['low_dim_state'] = data['obs']['low_dim_state']
            if 'low_dim_pcd' in data['obs']:
                new_obs['low_dim_pcd'] = data['obs']['low_dim_pcd'].unsqueeze(1)
            if 'rgb' in data['obs']:
                new_obs['rgb'] = data['obs']['rgb'].unsqueeze(1)
            if 'pcd' in data['obs']:
                new_obs['pcd'] = data['obs']['pcd'].unsqueeze(1)
            new_data['obs'] = new_obs

        return new_data
    
class Peract2Robomimic(ModuleAttrMixin):
    def __init__(self):
        super().__init__()
        self.robomimi2peract = Robomimic2Peract()

    def adapt(self, data):
        return self.robomimi2peract.unadapt(data)
    
    def unadapt(self, data):
        return self.robomimi2peract.adapt(data)



class WorldPoses2EEFPoses(ModuleAttrMixin):
    '''
    The following action adaptor receives the action as target poses in the world frame and converts it to displacements in the End-Effector frame.
    '''
    def __init__(self, add_global_pose=True):
        super(WorldPoses2EEFPoses, self).__init__()
        self.add_global_pose = add_global_pose

    def adapt(self, data):
        assert 'obs' in data
        obs = data['obs']

        ## End Effector Position (The following method assume that last instance in dimension 1 is the current end-effector pose) ##
        eef_pos = obs['robot0_eef_pos'][:, -1,:]
        eef_rot = obs['robot0_eef_rot'][:, -1, ...]
        ## Save End-Effector Position and Rotation ##
        self.eef_pos = eef_pos
        self.eef_rot = eef_rot

        inv_eef_pos, inv_eef_rot = se3_inverse(eef_pos, eef_rot)

        new_obs = {}
        for key in obs.keys():
            if key.find('rot') != -1:
                rot = obs[key]
                pos = obs[key.replace('rot', 'pos')]
                new_pos, new_rot = apply_transform(inv_eef_pos[:,None,:], inv_eef_rot[:,None,...], pos, rot)
                new_obs[key] = new_rot
                new_obs[key.replace('rot', 'pos')] = new_pos
            else:
                if key not in new_obs:
                    new_obs[key] = obs[key]

        if self.add_global_pose:
            new_obs['reference_pos'] = inv_eef_pos[:,None,:]
            new_obs['reference_rot'] = inv_eef_rot[:,None,...]

        if 'action' not in data:
            return {'obs': new_obs}

        else:
            action = data['action']
            ## Target Positions and Rotation ##
            act_p = action['act_p']
            act_r = action['act_r']
            act_gr = action['act_gr']
            act_ic = action['act_ic']
            act_p_ee, act_r_ee = apply_transform(inv_eef_pos[:, None, :], inv_eef_rot[:, None, :], act_p, act_r)
            new_action = {'act_p': act_p_ee, 'act_r': act_r_ee, 'act_gr': act_gr, 'act_ic': act_ic}

            return {'obs': new_obs, 'action': new_action}


    def unadapt(self, data):

        assert 'action' in data
        action = data['action']

        ## End Effector Position (The following method assume that last instance in dimension 1 is the current end-effector pose) ##
        eef_pos = self.eef_pos
        eef_rot = self.eef_rot

        ## Target Positions and Rotation ##
        act_p = action['act_p']
        act_r = action['act_r']
        act_gr = action['act_gr']
        if 'act_ic' in action:
            act_ic = action['act_ic']
        act_p_w, act_r_w = apply_transform(eef_pos[:,None,:], eef_rot[:,None,:], act_p, act_r)

        new_action = {'act_p': act_p_w, 'act_r': act_r_w, 'act_gr': act_gr}
        if 'act_ic' in action:
            new_action['act_ic'] = act_ic
        
        data['action'] = new_action
        return data


def test():
    from diffusion_policy.common.pytorch_util import print_dict, compare_dicts
    from diffusion_policy.model.common.so3_util import random_so3
    batch_size = 1
    num_obs_steps = 1
    batch = {
        'action': {
            'act_p': torch.randn(batch_size, num_obs_steps, 3),
            'act_r': random_so3(batch_size * num_obs_steps).reshape(batch_size, num_obs_steps, 3, 3),
            'act_gr': torch.randn(batch_size, num_obs_steps),
            'act_ic': torch.randn(batch_size, num_obs_steps)
        },
        'obs': {
            'robot0_eef_pos': torch.randn(batch_size, num_obs_steps, 3),
            'robot0_eef_rot': random_so3(batch_size * num_obs_steps).reshape(batch_size, num_obs_steps, 3, 3),
            'kp0_pos': torch.randn(batch_size, num_obs_steps, 3),
            'kp0_rot': random_so3(batch_size * num_obs_steps).reshape(batch_size, num_obs_steps, 3, 3),
            'kp1_pos': torch.randn(batch_size, num_obs_steps, 3),
            'kp1_rot': random_so3(batch_size * num_obs_steps).reshape(batch_size, num_obs_steps, 3, 3),

            'low_dim_state': torch.randn(batch_size, num_obs_steps, 3),
            'low_dim_pcd': torch.randn(batch_size, num_obs_steps, 3, 64, 64),
            'rgb': torch.randn(batch_size, num_obs_steps, 3, 64, 64),
            'pcd': torch.randn(batch_size, num_obs_steps, 3, 64, 64),
        },
        'extra': torch.randn(batch_size, num_obs_steps, 3)
    }

    adaptor = Robomimic2Peract()

    assert torch.allclose(batch['action']['act_r'], quaternion_to_matrix(matrix_to_quaternion(batch['action']['act_r'])), atol=1e-3)

    print_dict(batch)
    print_dict(adaptor.adapt(batch))
    compare_dicts(batch, adaptor.unadapt(adaptor.adapt(batch)))


    # adaptor = SequenceAdaptor(WorldPoses2EEFPoses(), Robomimic2Peract())
    # print_dict(batch['action'], print_values=True)
    # print_dict(adaptor.adapt(batch))

if __name__ == "__main__":
    test()
    print("All tests passed!")