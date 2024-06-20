
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
                use_rgb = False,
                use_mask = False,
                use_pcd = False,
                use_depth = False,
                use_low_dim_pcd = False,
                use_pose = False,
                use_low_dim_state = False,
                channels_last: bool = False):
    grip_mat = obs.gripper_matrix
    low_dim_pcd = obs.misc.get('low_dim_pcd', None)
    keypoint_poses = obs.misc.get('low_dim_poses', None)
    # obs.gripper_pose = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    obs.joint_velocities = None
    obs.task_low_dim_state = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)
        
    remove_keys =  ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose', 'gripper_matrix', 'ignore_collisions',
               'gripper_joint_positions', 'gripper_touch_forces', 'misc']
    
    if use_rgb is False:
        for camera in cameras:
            obs.misc['%s_rgb' % camera] = None
            obs.__dict__['%s_rgb' % camera] = None
    if use_mask is False:
        for camera in cameras:
            obs.misc['%s_mask' % camera] = None
            obs.__dict__['%s_mask' % camera] = None
    if use_depth is False:
        for camera in cameras:
            obs.misc['%s_depth' % camera] = None
            obs.__dict__['%s_depth' % camera] = None
    if use_pcd is False:
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

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    obs_dict['robot0_eef_rot'] = grip_mat[...,:3,:3]
    obs_dict['robot0_eef_pos'] = grip_mat[...,:3,3]

    obs_dict['curr_gripper'] = pose_gr_ic_from_obs(obs)

    if use_rgb:
        obs_dict['rgb'] = np.stack([obs_dict.pop('%s_rgb' % camera) for camera in cameras]) 
    if use_pcd:
        obs_dict['pcd'] = np.stack([obs_dict.pop('%s_point_cloud' % camera) for camera in cameras])
    if use_low_dim_pcd:
        obs_dict['low_dim_pcd'] = low_dim_pcd
    if use_pose:
        for i, kp in enumerate(keypoint_poses):
            obs_dict[f'kp{i}_pos'] = kp[:3,3]
            obs_dict[f'kp{i}_rot'] = kp[:3,:3]
    if use_low_dim_state:
        obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    return obs_dict

# taken from https://github.com/stepjam/ARM/blob/main/arm/utils.py
def normalize_quaternion(quat):
    quat = np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
    if quat[-1] < 0:
        quat = -quat
    return quat

def quaternion_to_real_part_first(quat : np.ndarray):
    """
    Rearranges the quaternion array to have the real part as the first element.

    Parameters:
    quat (np.ndarray): The input quaternion array.

    Returns:
    np.ndarray: The rearranged quaternion array with the real part as the first element.
    """
    return np.concatenate([quat[..., [3]], quat[..., :3]], axis=-1)

def quaternion_to_real_part_last(quat):
    """
    Converts a quaternion array to a new array with the real part at the last index.

    Args:
        quat (numpy.ndarray): Array of quaternions.

    Returns:
        numpy.ndarray: Array with the real part of each quaternion at the last index.

    """
    return np.concatenate([quat[..., 1:], quat[..., [0]]], axis=-1)

def stack_list_of_dicts(data):
    data_dict = {}
    for k in data[0].keys():
        if isinstance(data[0][k], (np.ndarray, np.float32, int, np.bool_)):
            data_dict[k] = np.stack([d[k] for d in data])
        else:
            data_dict[k] = stack_list_of_dicts([d[k] for d in data])
    return data_dict

def pose_gr_ic_from_obs(obs, obs_m1=None):
    """
    Convert observation data to a concatenated array representing the gripper pose, gripper open state, and ignore collisions flag.

    Args:
        obs (object): The current observation object.
        obs_m1 (object, optional): The previous observation object. Defaults to None.

    Returns:
        numpy.ndarray: A concatenated array representing the gripper pose, gripper open state, and ignore collisions flag.
    """
    obs_m1 = obs if obs_m1 is None else obs_m1

    ignore_collisions = np.array(obs_m1.ignore_collisions, dtype=np.float32).reshape(1)
    gripper_open = np.array(obs.gripper_open, dtype=np.float32).reshape(1)
    
    gripper_pose = obs.gripper_pose
    pos = gripper_pose[:3]
    quat = gripper_pose[3:]
    quat = normalize_quaternion(quaternion_to_real_part_first(quat))
    return np.concatenate([pos, quat, gripper_open, ignore_collisions])


def create_actions_from_obs(observations):
    actions = {
        'act_p': [],
        'act_r': [],
        'act_gr': [],
        'act_ic': [],
        'gt_trajectory': []
    }
    for i in range(1, len(observations)):
        obs_tp1 = observations[i]
        obs_tm1 = observations[i - 1]
        ignore_collisions = np.array(obs_tm1.ignore_collisions, dtype=np.float32).reshape(1)
        gripper_open = np.array(obs_tp1.gripper_open, dtype=np.float32).reshape(1)
        action = dict()
        action['act_p'] = obs_tp1.gripper_matrix[:3,3].reshape(-1, 3)
        action['act_r'] = obs_tp1.gripper_matrix[:3,:3].reshape(-1, 3, 3)
        action['act_gr'] = gripper_open
        action['act_ic'] = ignore_collisions

        action['gt_trajectory'] = pose_gr_ic_from_obs(obs_tp1, obs_tm1).reshape(-1, 9)

        for k in actions.keys():
            actions[k].append(action[k])

    actions = {k: np.concatenate(v) for k, v in actions.items()}
    return actions

def create_obs_action(demo, 
                  curr_obs_idx,
                  use_keyframe_actions=True,
                  use_keyframe_observations=False,
                  horizon=1,
                  episode_keypoints=None, 
                  curr_kp_idx=0, 
                  cameras=[], 
                  n_obs=2, 
                  use_low_dim_pcd=False, 
                  use_pcd=False, 
                  use_rgb=False, 
                  use_mask=False,
                  use_depth=False,
                  use_pose=False, 
                  use_low_dim_state=False,
                  obs_augmentation_every_n=10):
    if use_keyframe_observations:
        observations = []
        for i in reversed(range(0, n_obs)):
            offset = i
            if curr_obs_idx == episode_keypoints[curr_kp_idx - 1]:
                offset +=1
            if curr_kp_idx - offset < 0:
                observations.append(demo[0])
            else:
                observations.append(demo[episode_keypoints[curr_kp_idx - offset]])
    else:
        observations = [demo[max(0, curr_obs_idx - i * obs_augmentation_every_n)] for i in reversed(range(n_obs))]
    obs_dicts = None
    for _, obs in enumerate(observations):
        obs_dict = extract_obs(obs, cameras=cameras, use_rgb=use_rgb, use_mask=use_mask, use_pcd=use_pcd, use_depth=use_depth, use_low_dim_pcd=use_low_dim_pcd, use_pose=use_pose, use_low_dim_state=use_low_dim_state)
        if obs_dicts is None:
            obs_dicts = {k: [] for k in obs_dict.keys()}
        for k in obs_dict.keys():
            obs_dicts[k].append(obs_dict[k])

    obs_dicts = {k: np.stack(v) for k, v in obs_dicts.items()}

    if use_keyframe_actions:
        obs_tp1 = demo[episode_keypoints[curr_kp_idx]]
        obs_tm1 = demo[episode_keypoints[max(0,curr_kp_idx - 1)]]
        obs2action_list = [obs_tm1, obs_tp1]
    else:
        obs2action_list = [observations[-1]]
        obs2action_list.extend([demo[min(len(demo)-1, curr_obs_idx + (i + 1))] for i in range(horizon)])
    action_dicts = create_actions_from_obs(obs2action_list)

    data = {
        'obs': obs_dicts,
        'action': action_dicts
    }

    return data

def create_dataset(demos, cameras, demo_augmentation_every_n=10, 
                   obs_augmentation_every_n=10, n_obs=2, 
                   use_low_dim_pcd=False, use_rgb=False, 
                    use_keyframe_actions=True, horizon=1,
                   use_pcd=False, use_mask=False, use_depth=False,
                    use_pose=False, use_low_dim_state=False, use_keyframe_observations=False):
    dataset = []
    episode_begin = [0]

    if use_keyframe_observations:
        demo_augmentation_every_n = 1

    for demo in demos:
        episode_keypoints = _keypoint_discovery(demo)
        episode_keypoints = [0] + episode_keypoints
        next_keypoint = 1
        i = 0
        while i < len(demo)-1:
            if i % demo_augmentation_every_n:
                i += 1
                continue
            
            if use_keyframe_actions:
                while next_keypoint < len(episode_keypoints) and i >= episode_keypoints[next_keypoint]:
                    next_keypoint += 1
                if i >= episode_keypoints[-1]:
                    break            
            data = create_obs_action(demo=demo, curr_obs_idx=i, episode_keypoints=episode_keypoints, curr_kp_idx=next_keypoint, cameras=cameras, n_obs=n_obs, use_keyframe_actions=use_keyframe_actions, horizon=horizon,
                                    use_low_dim_pcd=use_low_dim_pcd, use_pcd=use_pcd, use_rgb=use_rgb, use_pose=use_pose, use_depth=use_depth,
                                    use_mask=use_mask, use_low_dim_state=use_low_dim_state, obs_augmentation_every_n=obs_augmentation_every_n,
                                    use_keyframe_observations=use_keyframe_observations)
            dataset.append(data)
            episode_begin.append(episode_begin[-1])
            i += 1

            if use_keyframe_observations:
                i = episode_keypoints[next_keypoint]

        episode_begin[-1] = len(dataset)
    return dataset, episode_begin