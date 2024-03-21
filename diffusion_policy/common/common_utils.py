import pickle
from typing import Dict, Optional, Sequence
from pathlib import Path
import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation

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
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


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
    rot = torch.from_numpy(
        Rotation.from_quat(quat.reshape(-1,4).cpu()).as_matrix().reshape(B, T, 3, 3)
        ).to(device=device, dtype=dtype)
    
    trajectory = torch.eye(4).unsqueeze(0).repeat(B, T, 1, 1).to(device=device, dtype=dtype)
    trajectory[:, :, :3, :3] = rot
    trajectory[:, :, :3, 3] = pos

    gripper_open = action[:, :, 7]
    ignore_collision = action[:, :, 8]

    return trajectory, gripper_open, ignore_collision

def action_from_trajectory_gripper_open_ignore_collision(trajectory : torch.Tensor, gripper_open : torch.Tensor, ignore_collision : torch.Tensor):
    B, T, _, _ = trajectory.shape
    device, dtype = trajectory.device, trajectory.dtype
    gripper_open = gripper_open.reshape(B, T, 1)
    ignore_collision = ignore_collision.reshape(B, T, 1)

    pos = trajectory[:, :, :3, 3]
    rot = trajectory[:, :, :3, :3]
    quat = torch.from_numpy(
        Rotation.from_matrix(rot.reshape(-1,3,3).cpu()).as_quat().reshape(B, T, 4)
        ).to(device=device, dtype=dtype)

    action = torch.cat([pos, quat, gripper_open, ignore_collision], dim=-1)
    return action
    