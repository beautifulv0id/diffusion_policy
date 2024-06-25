from pathlib import Path
import json
from typing import List,Dict, Any
import numpy as np
import random
import torch
from rlbench.demo import Demo
from pyrep.const import ObjectType
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from diffusion_policy.common.rlbench_util import create_rlbench_action
from diffusion_policy.common.rlbench_util import _keypoint_discovery



ALL_RLBENCH_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk', 
    'open_drawer_keypoint', 'stack_blocks'
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action, collision_checking=False):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        action_dim=8,
    ):
        self._policy = policy
        self._instructions = instructions
        self._action_dim = action_dim

        self._actions = {}
        self._instr = None
        self._task_str = None
        self._task_id = None

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        # TODO: bring back
        # instructions = list(self._instructions[task_str][variation])
        # self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = _keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, obs):
        """
        Args:
            obs: dict
        Returns:
            action: torch.Tensor
        """
        # self._task_id = self._task_id.to(self.device)
        pred_dict = self._policy.predict_action(obs)
    
        if "rlbench_action" in pred_dict:
            return {
                "rlbench_action": pred_dict["rlbench_action"],
            }
        rot = pred_dict['action']['act_r']
        pos = pred_dict['action']['act_p']
        gripper_open = pred_dict['action'].get('act_gr', None)
        ignore_collision = pred_dict['action'].get('act_ic', None)
        rlbench_action = create_rlbench_action(rot, pos, gripper_open, ignore_collision)
        rlbench_action = rlbench_action[0]
        out = {
            "rlbench_action": rlbench_action.detach().cpu().numpy(),
            # "action": pred_dict['action'],
        }
        return out

    @property
    def device(self):
        return next(self._policy.parameters()).device
    
    @property
    def dtype(self):
        return next(self._policy.parameters()).dtype


def get_object_pose_indices_from_task(task : Task):
    task.set_initial_objects_in_scene()
    state_sizes = []
    mask = []
    for obj, objtype in task._initial_objs_in_scene:
        valid = True
        state_size = 7
        if objtype != ObjectType.SHAPE:
            valid = False
        if objtype == ObjectType.JOINT:
            state_size += 1
        elif objtype == ObjectType.FORCE_SENSOR:
            state_size += 6
            valid = False
        state_sizes.append(state_size)
        mask.append(valid)
    state_sizes = np.array(state_sizes)
    state_idxs = state_sizes[mask]
    return state_idxs

def get_actions_from_demo(demo):
    """
    Fetch the desired state and action based on the provided demo.
        :param demo: fetch each demo and save key-point observations
        :return: a list of obs and action
    """
    key_frame = _keypoint_discovery(demo)

    action_ls = []
    for i in range(len(key_frame)):
        obs = demo[key_frame[i]]
        action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        action = torch.from_numpy(action_np)
        action_ls.append(action.unsqueeze(0))
    return action_ls

class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam
        self.save_pose()

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(0)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])
