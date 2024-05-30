
from rlbench.demo import Demo
from typing import List
from rlbench.backend.observation import Observation
import numpy as np
REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces']

CAMERAS = ['left_shoulder', 'right_shoulder', 'wrist', 'overhead', 'front']

def _is_stopped(demo : Demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo: Demo,
                        stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    # print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def extract_obs(obs: Observation,
				cameras,
                mask = False,
                depth = False,
                pcd = False,
                channels_last: bool = False):
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    object_poses = obs.misc.get('object_poses', None)
    low_dim_pcd = obs.misc.get('low_dim_pcd', None)
    keypoint_poses = obs.misc.get('low_dim_poses', None)
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    obs.joint_velocities = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)
        
    remove_keys =  ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces']
    
    if mask is False:
        for camera in cameras:
            obs.misc['%s_mask' % camera] = None
            obs.__dict__['%s_mask' % camera] = None
    if depth is False:
        for camera in cameras:
            obs.misc['%s_depth' % camera] = None
            obs.__dict__['%s_depth' % camera] = None
    if pcd is False:
        for camera in cameras:
            obs.misc['%s_point_cloud' % camera] = None
            obs.__dict__['%s_point_cloud' % camera] = None

    remove_cameras = [cam for cam in CAMERAS if cam not in cameras] 
    for camera in remove_cameras:
        obs.misc['%s_camera_extrinsics' % camera] = None
        obs.misc['%s_camera_intrinsics' % camera] = None
        obs.__dict__['%s_mask' % camera] = None
        obs.__dict__['%s_depth' % camera] = None
        obs.__dict__['%s_point_cloud' % camera] = None
        obs.__dict__['%s_rgb' % camera] = None

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([
                  obs.gripper_open,
                  *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in remove_keys}
    
    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if isinstance(v, np.ndarray) and v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)


    for camera_name in cameras:
          obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
          obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    obs_dict['gripper_matrix'] = obs.gripper_matrix
    obs_dict['gripper_pose'] = obs.gripper_pose
    obs_dict['object_poses'] = object_poses
    obs_dict['low_dim_pcd'] = low_dim_pcd
    obs_dict['keypoint_poses'] = keypoint_poses
    return obs_dict

# taken from https://github.com/stepjam/ARM/blob/main/arm/utils.py
def normalize_quaternion(quat):
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    if quat[-1] < 0:
        quat = -quat
    return quat

def create_sample(demo, 
                  episode_keypoints, 
                  curr_kp_idx, 
                  cameras, 
                  n_obs=2, 
                  use_low_dim_pcd=False, 
                  use_pcd=False, 
                  use_rgb=False, 
                  use_pose=False, 
                  use_low_dim_state=False,
                  obs_augmentation_every_n=10):
    curr_obs_idx = episode_keypoints[max(0, curr_kp_idx - 1)]
    observations = [demo[max(0, curr_obs_idx - i * obs_augmentation_every_n)] for i in range(n_obs)]
    obs_tp1 = demo[episode_keypoints[curr_kp_idx]]
    obs_tm1 = demo[episode_keypoints[curr_kp_idx - 1]]
    ignore_collisions = np.array(obs_tm1.ignore_collisions, dtype=np.float32)
    gripper_open = np.array(obs_tp1.gripper_open, dtype=np.float32)
    action = np.concatenate([obs_tp1.gripper_pose[:3], normalize_quaternion(obs_tp1.gripper_pose[3:]), [gripper_open], [ignore_collisions]])
    data = []
    for _, obs in enumerate(observations):
        full_obs_dict = extract_obs(obs, cameras, pcd=use_pcd)
        obs_dict = {}
        obs_dict['robot0_eef_rot'] = full_obs_dict['gripper_matrix'][...,:3,:3]
        obs_dict['robot0_eef_pos'] = full_obs_dict['gripper_matrix'][...,:3,3]
        if use_rgb:
            obs_dict['rgb'] = np.stack([full_obs_dict['%s_rgb' % camera] for camera in cameras]) 
        if use_pcd:
            obs_dict['pcd'] = np.stack([full_obs_dict['%s_point_cloud' % camera] for camera in cameras])
        if use_low_dim_pcd:
            obs_dict['low_dim_pcd'] = full_obs_dict['low_dim_pcd']
        if use_pose:
            for i, kp in enumerate(full_obs_dict['keypoint_poses']):
                obs_dict[f'kp{i}_pos'] = kp[:3,3]
                obs_dict[f'kp{i}_rot'] = kp[:3,:3]
        if use_low_dim_state:
            obs_dict['low_dim_state'] = full_obs_dict['low_dim_state']
        data.append({
            "obs": obs_dict,
        })
    
    def stack_list_of_dicts(data):
        data_dict = {}
        for k in data[0].keys():
            if isinstance(data[0][k], (np.ndarray, np.float32, int, np.bool_)):
                data_dict[k] = np.stack([d[k] for d in data])
            else:
                data_dict[k] = stack_list_of_dicts([d[k] for d in data])
        return data_dict
    data = stack_list_of_dicts(data)
    data['action'] = dict()
    data['action']['act_p'] = obs_tp1.gripper_matrix[:3,3].reshape(-1, 3)
    data['action']['act_r'] = obs_tp1.gripper_matrix[:3,:3].reshape(-1, 3, 3)
    data['action']['act_gr'] = gripper_open.reshape(1)
    data['action']['act_ic'] = ignore_collisions.reshape(1)

    return data

def create_dataset(demos, cameras, demo_augmentation_every_n=10, 
                   obs_augmentation_every_n=10, n_obs=2, 
                   use_low_dim_pcd=False, use_rgb=False, use_pcd=False, 
                   keypoints_only=False, use_pose=False, use_low_dim_state=False):
    dataset = []
    episode_begin = [0]

    for demo in demos:
        episode_keypoints = _keypoint_discovery(demo)
        if keypoints_only:
            for i in range(1, len(episode_keypoints)):
                data = create_sample(demo, episode_keypoints, i, cameras, n_obs, 
                                     use_low_dim_pcd, use_pcd, use_rgb, use_pose, 
                                     use_low_dim_state, obs_augmentation_every_n)
                dataset.append(data)
                episode_begin.append(episode_begin[-1])
            episode_begin[-1] = len(dataset)
        else:
            next_keypoint = 0
            for i in range(episode_keypoints[0], len(demo)-1):
                if i % demo_augmentation_every_n:
                    continue

                while next_keypoint < len(episode_keypoints) and i >= episode_keypoints[next_keypoint]:
                    next_keypoint += 1
                if i >= episode_keypoints[-1]:
                    break            
                data = create_sample(demo, episode_keypoints, next_keypoint, cameras, n_obs, 
                                     use_low_dim_pcd, use_pcd, use_rgb, use_pose, 
                                     use_low_dim_state, obs_augmentation_every_n)
                dataset.append(data)
                episode_begin.append(episode_begin[-1])
            episode_begin[-1] = len(dataset)
    return dataset, episode_begin