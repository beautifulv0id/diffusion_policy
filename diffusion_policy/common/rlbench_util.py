
from rlbench.demo import Demo
from typing import List
from rlbench.backend.observation import Observation
import numpy as np
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode
from torch.nn.functional import interpolate
import torch
from torch import einsum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, standardize_quaternion
import pickle
from typing import Dict, Optional, Sequence
from pathlib import Path
import json
from PIL import Image
import io
import einops
import torch.nn.functional as F


REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces']

CAMERAS = ['left_shoulder', 'right_shoulder', 'wrist', 'overhead', 'front']

Instructions = Dict[str, Dict[int, torch.Tensor]]


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    return gripper_loc_bounds

def get_task_num_objects(path: str, task: str):
    num_objects = json.load(open(path, "r"))
    return num_objects[task]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None


def trajectory_gripper_open_ignore_collision_from_action(action : torch.Tensor):
    B, T, _ = action.shape
    device, dtype = action.device, action.dtype

    pos = action[:, :, :3]
    quat = action[:, :, 3:7]
    rot = quaternion_to_matrix(torch.cat([quat[..., [3]], quat[..., :3]], dim=-1))   
    trajectory = torch.eye(4).unsqueeze(0).repeat(B, T, 1, 1).to(device=device, dtype=dtype)
    trajectory[:, :, :3, :3] = rot
    trajectory[:, :, :3, 3] = pos

    gripper_open = action[:, :, 7]
    ignore_collision = action[:, :, 8]

    return trajectory, gripper_open, ignore_collision

def action_from_trajectory_gripper_open_ignore_collision(trajectory : torch.Tensor, gripper_open : torch.Tensor, ignore_collision : torch.Tensor):
    B, T, _, _ = trajectory.shape
    device, dtype = trajectory.device, trajectory.dtype
    gripper_open = gripper_open.reshape(B, T, 1) > 0.5
    ignore_collision = ignore_collision.reshape(B, T, 1) > 0.5

    pos = trajectory[:, :, :3, 3]
    rot = trajectory[:, :, :3, :3]
    quat = standardize_quaternion(matrix_to_quaternion(rot))
    quat = torch.cat([quat[..., 1:], quat[..., [0]]], dim=-1)
    action = torch.cat([pos, quat, gripper_open, ignore_collision], dim=-1)
    return action

def create_rlbench_action(rot: torch.Tensor, pos: torch.Tensor, gripper_open: torch.Tensor = None, ignore_collision: torch.Tensor = None):
    B, T, _ = pos.shape
    quat = standardize_quaternion(matrix_to_quaternion(rot))
    quat = torch.cat([quat[..., 1:4], quat[..., [0]]], dim=-1)
    action = torch.cat([pos, quat], dim=-1)
    if gripper_open is not None:
        gripper_open = gripper_open.reshape(B, T, 1) > 0.5
        action = torch.cat([action, gripper_open], dim=-1)
    if ignore_collision is not None:
        ignore_collision = ignore_collision.reshape(B, T, 1) > 0.5
        action = torch.cat([action, ignore_collision], dim=-1)
    return action

def create_robomimic_from_rlbench_action(rlbench_action: torch.Tensor, rel_last: bool = True):
    L = rlbench_action.shape[-1]
    pos = rlbench_action[:, :, :3]
    quat = rlbench_action[:, :, 3:7]
    if rel_last:
        quat = torch.cat([quat[..., [3]], quat[..., :3]])
    rot = quaternion_to_matrix(quat)
    action = {
            "act_r": rot,
            "act_p": pos
        }
    if L > 7:
        gripper_open = rlbench_action[:, :, 7]
        action["act_gr"] = gripper_open
    if L > 8:
        ignore_collision = rlbench_action[:, :, 8]
        action["act_ic"] = ignore_collision
    return action

def mask_out_features_pcd(mask, rgbs, pcd, n_min = 0, n_max=1000000):
    """
    Masks out features and point cloud data based on a given mask.

    Args:
        mask (torch.Tensor): (B, ncam, 1, H, W)
        rgb (torch.Tensor): (B, ncam, F, H, W)
        pcd (torch.Tensor): (B, ncam, 3, H, W)
        n_min (int, optional): 
        n_max (int, optional): 
    Returns:
        rgb (torch.Tensor): (B, N, F)
        pcd (torch.Tensor): (B, N, 3)
    """
    this_mask = mask.clone()
    b, v, _, h, w = rgbs.shape
    this_mask = F.interpolate(this_mask.flatten(0,1).float(), (h, w), mode='nearest').bool().reshape(b, v, h, w)

    B = this_mask.size(0)
    n = this_mask.view(B, -1).count_nonzero(dim=-1)
    n_sample = torch.clamp(n.max(), n_min, n_max)
    diff = n_sample - n
    neg_inds = (~this_mask.view(B,-1)).nonzero(as_tuple=True)[1]
    neg_indsn = (~this_mask.view(B,-1)).count_nonzero(dim=-1)
    neg_indsn = torch.cat([torch.zeros(1, device=mask.device), torch.cumsum(neg_indsn, dim=0)])
    idx0 = torch.tensor([], device=mask.device, dtype=torch.int)
    idx1 = torch.tensor([], device=mask.device, dtype=torch.int)
    for i in range(B):
        offset = diff[i].int().item()
        if offset > 0:
            neg_i = neg_indsn[i].int().item()
            idx0 = torch.cat((idx0, torch.full((offset,), i, device=mask.device)))
            idx1 = torch.cat((idx1, neg_inds[neg_i:neg_i+offset]))
    fill_inds = (idx0, idx1)
    this_mask.view(B, -1)[fill_inds] = True
    idx = this_mask.view(B, -1).nonzero(as_tuple=True)

    pos_inds = this_mask.view(B, -1).nonzero(as_tuple=True)[1]
    pos_indsn = this_mask.view(B, -1).count_nonzero(dim=-1)
    pos_indsn = torch.cat([torch.zeros(1, device=mask.device), torch.cumsum(pos_indsn, dim=0)])
    idx0 = torch.tensor([], device=mask.device, dtype=torch.int)
    idx1 = torch.tensor([], device=mask.device, dtype=torch.int)
    for i in range(B):
        offset = -diff[i].int().item()
        if offset > 0:
            pos_i = pos_indsn[i].int().item()
            idx0 = torch.cat((idx0, torch.full((offset,), i, device=mask.device)))
            idx1 = torch.cat((idx1, pos_inds[pos_i:pos_i+offset]))

    fill_inds = (idx0, idx1)
    this_mask.view(B, -1)[fill_inds] = False
    idx = this_mask.view(B, -1).nonzero(as_tuple=True)

    rgbs = einops.rearrange(rgbs, 'b v c h w -> b (v h w) c')
    rgbs = rgbs[idx].reshape(B, n_sample, -1)

    pcd = einops.rearrange(pcd, 'b v c h w -> b (v h w) c')
    pcd = pcd[idx].reshape(B, n_sample, -1)

    return rgbs, pcd

def create_obs_state_plot(obs, action=None, downsample=1, use_mask=False):
    pcd = obs['pcd']
    rgb = obs['rgb']
    mask = obs['mask'] if use_mask else None
    curr_gripper = obs['curr_gripper']
    out = create_robomimic_from_rlbench_action(curr_gripper, rel_last=False)
    curr_gripper_rot = out['act_r'].squeeze(0).float()
    curr_gripper_pos = out['act_p'].squeeze(0).float()

    n_obs = curr_gripper_rot.shape[0]

    colors = cm.get_cmap('Reds',  n_obs+ 1)

    def create_gripper_pts(scale = 0.2):
        gripper_pts = torch.tensor([
            [0, 0, -1],
            [0, 0, 0],
            [0, -0.5, 0],
            [0, 0.5, 0],
            [0, -0.5, 0],
            [0, -0.5, 0.5],
            [0, 0.5, 0],
            [0, 0.5, 0.5],
            
        ]) 

        gripper_pts = gripper_pts + torch.tensor([0, 0, -1])
        gripper_pts = gripper_pts * scale

        return gripper_pts
    
    def plot_gripper(ax, gripper_pos, gripper_rot, scale = 0.2):
        gripper_pts = create_gripper_pts(scale)
        gripper_pts = einsum('nij,kj->nki', gripper_rot, gripper_pts)
        gripper_pts = gripper_pts + gripper_pos.unsqueeze(1)
        gripper_pts = gripper_pts.reshape(gripper_pos.shape[0], -1, 3).numpy()
        for i, gripper_pts_i in enumerate(gripper_pts):
            for j in range(0, len(gripper_pts_i), 2):
                line = ax.plot(gripper_pts_i[j:j+2,0], gripper_pts_i[j:j+2,1], gripper_pts_i[j:j+2,2], linewidth=1, zorder=10, color=colors(i+1))
                if j == 0:
                    line[0].set_label(f"Gripper t={i-n_obs+1}")

    def plot_pcd(ax, pcd, rgb, mask=None):
        if downsample > 1:
            b, v, _, h, w = rgb.shape
            h, w = rgb.shape[-2] // downsample, rgb.shape[-1] // downsample

            pcd = interpolate(pcd.reshape((-1,) + pcd.shape[-3:]), size=(h, w), mode='bilinear').reshape(b, v, 3, h, w)
            rgb = interpolate(rgb.reshape((-1,) + rgb.shape[-3:]), size=(h, w), mode='bilinear').reshape(b, v, 3, h, w)
            if mask is not None:
                mask = interpolate(rgb.reshape((-1,) + rgb.shape[-3:]).float(), size=(h, w), mode='nearest').bool().reshape(b, v, 1, h, w)

        if mask is not None:
            rgb, pcd = mask_out_features_pcd(mask, rgb, pcd, n_min=1, n_max=1000000)
            rgb = rgb.flatten(0,1)
            pcd = pcd.flatten(0,1)
        else:
            pcd = pcd.permute(0, 1, 3, 4, 2).reshape(-1, 3)
            rgb = rgb.permute(0, 1, 3, 4, 2).reshape(-1, 3)

        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c=rgb, s=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45, roll=0)
    RADIUS = .7  # Control this value.
    ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_zlim3d(-RADIUS / 2 + 1, RADIUS / 2 + 1)
    ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
    plot_pcd(ax, pcd, rgb, mask)
    plot_gripper(ax, curr_gripper_pos, curr_gripper_rot, scale=0.1)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf)).transpose(2, 0, 1)
    buf.close()

    # Image.fromarray(image).save('obs.png')
    return image

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

def remove_cameras_from_obs(obs, cameras):
    for camera in cameras:
        obs.misc['%s_camera_extrinsics' % camera] = None
        obs.misc['%s_camera_intrinsics' % camera] = None
        obs.__dict__['%s_mask' % camera] = None
        obs.__dict__['%s_depth' % camera] = None
        obs.__dict__['%s_point_cloud' % camera] = None
        obs.__dict__['%s_rgb' % camera] = None


def extract_obs(obs: Observation,
				cameras,
                use_rgb = False,
                use_mask = False,
                use_pcd = False,
                use_depth = False,
                use_low_dim_pcd = False,
                use_pose = False,
                use_low_dim_state = False,
                channels_last: bool = False,
                mask_ids=None):
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

    remove_cameras_from_obs(obs, [cam for cam in CAMERAS if cam not in cameras] )

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
        rgb = np.stack([obs_dict.pop('%s_rgb' % camera) for camera in cameras])
        obs_dict['rgb'] = rgb / 255.0
    if use_pcd:
        obs_dict['pcd'] = np.stack([obs_dict.pop('%s_point_cloud' % camera) for camera in cameras])
    if use_mask:
        mask = np.stack([obs_dict.pop('%s_mask' % camera) for camera in cameras])
        # for m in mask:
            # m = m.transpose(1, 2, 0).squeeze()
            # print("min", np.min(m), "max", np.max(m))
            # def mask_to_rgb_handles(mask):
            #     mask = mask.reshape(mask.shape[0], mask.shape[1])
            #     # mask should be (w, h)
            #     r = mask % 256
            #     g = (mask // 256)
            #     b = (mask // (256 * 256))
            #     return np.stack([r, g, b], axis=2)
            # m = mask_to_rgb_handles(m).astype(np.uint8)
            # # print(np.unique(m))
            # mask_img = Image.fromarray(m)
            # mask_img.save(f"/home/felix/Workspace/diffusion_policy_felix/0.png")
        mask = (mask > mask_ids[0]) & (mask < mask_ids[1])
        obs_dict['mask'] = mask.astype(np.bool)
        # obs_dict['obj_inds'] = obj_ids # TODO: remove this

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

def create_obs_config(
    image_size, apply_rgb, apply_depth, apply_pc, apply_mask, apply_cameras, **kwargs
):
    """
    Set up observation config for RLBench environment.
        :param image_size: Image size.
        :param apply_rgb: Applying RGB as inputs.
        :param apply_depth: Applying Depth as inputs.
        :param apply_pc: Applying Point Cloud as inputs.
        :param apply_cameras: Desired cameras.
        :return: observation config
    """
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=apply_rgb,
        point_cloud=apply_pc,
        depth=apply_depth,
        mask=apply_mask,
        image_size=image_size,
        render_mode=RenderMode.OPENGL,
        **kwargs,
    )

    camera_names = apply_cameras
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams

    obs_config = ObservationConfig(
        front_camera=kwargs.get("front", unused_cams),
        left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        wrist_camera=kwargs.get("wrist", unused_cams),
        overhead_camera=kwargs.get("overhead", unused_cams),
        joint_forces=False,
        joint_positions=False,
        joint_velocities=True,
        task_low_dim_state=True,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )

    return obs_config


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
                  mask_ids=None,
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
        obs_dict = extract_obs(obs, cameras=cameras, use_rgb=use_rgb, use_mask=use_mask, use_pcd=use_pcd, use_depth=use_depth, use_low_dim_pcd=use_low_dim_pcd, use_pose=use_pose, use_low_dim_state=use_low_dim_state, mask_ids=mask_ids)
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
                   use_pcd=False, use_mask=False, mask_ids=None,use_depth=False,
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
                                    use_keyframe_observations=use_keyframe_observations, mask_ids=mask_ids)
            dataset.append(data)
            episode_begin.append(episode_begin[-1])
            i += 1
        episode_begin[-1] = len(dataset)
        for obs in demo:
            remove_cameras_from_obs(obs, cameras)
    return dataset, episode_begin