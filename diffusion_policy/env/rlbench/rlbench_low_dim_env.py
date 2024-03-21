import os
import glob
import random
from typing import List, Dict, Any
from pathlib import Path
import json

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from diffusion_policy.common.launch_utils import low_dim_obs_to_keypoints
from scipy.spatial.transform import Rotation as R


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
    'water_plants', 'wipe_desk'
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
        self._gripper_bounds = np.array(        [[
            0.10261330753564835,
            -0.06472617387771606,
            0.9140162467956543
        ],
        [
            0.3657068908214569,
            0.4357367157936096,
            1.2285298109054565
        ]]
)

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

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

    def predict(self, keypoint_idxs, keypoints, gripper):
        """
        Args:
            keypoint_idxs: (bs, num_hist, N)
            keypoints: (bs, num_hist, N, 3)
            gripper: (B, nhist, 4, 4)

        Returns:
            action: torch.Tensor
        """

        self._task_id = self._task_id.to(gripper.device)
        obs = {
            "keypoint_idx": keypoint_idxs,
            "keypoint_pcd": keypoints,
            "agent_pose": gripper,
        }
        action = self._policy.predict_action(obs)["action"][0,0,:self._action_dim]
        pos = action[:3]

        # apply gripper bounds
        for i in range(3):
            if pos[i] < self._gripper_bounds[0][i] or pos[i] > self._gripper_bounds[1][i]:
                print(f"Clipping pos {i} from {pos[i]} to {self._gripper_bounds[0][i]}")
            pos[i] = np.clip(pos[i], self._gripper_bounds[0][i], self._gripper_bounds[1][i])

        return action.detach().cpu().numpy()

    @property
    def device(self):
        return next(self._policy.parameters()).device


class RLBenchLowDimEnv:

    def __init__(
        self,
        data_path,
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=10,
    ):

        # setup required inputs
        self.data_path = data_path

        # setup RLBench environments
        self.obs_config = self.create_obs_config()

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.obs_history = []
        self.obs_history_augmentation_every_n = obs_history_augmentation_every_n
        self.action_mode.arm_action_mode.set_callable_each_step(lambda obs: self.obs_history.append(obs))
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )


    def get_keypoints_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        pcd = torch.from_numpy(low_dim_obs_to_keypoints(obs.task_low_dim_state))
        keypoint_idx = torch.arange(pcd.shape[0])
        agent_pose = torch.from_numpy(obs.gripper_matrix)
        return pcd, keypoint_idx, agent_pose
    
    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=100,
        num_history=1,
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = task.variation_count()

        if num_variations > 0:
            task_variations = np.minimum(num_variations, task_variations)
            task_variations = range(task_variations)
        else:
            task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    num_demos=num_demos // len(task_variations) + 1,
                    actioner=actioner,
                    max_tries=max_tries,
                    verbose=verbose,
                    dense_interpolation=dense_interpolation,
                    interpolation_length=interpolation_length,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=50,
        num_history=0,
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        for demo_id in range(num_demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                num_valid_demos += 1
            except:
                continue

            keypoint_idxs = torch.Tensor([]).to(device)
            keypoints = torch.Tensor([]).to(device)
            grippers = torch.Tensor([]).to(device)

            # descriptions, obs = task.reset()
            descriptions, obs = task.reset_to_demo(demo)
            self.obs_history.append(obs)

            actioner.load_episode(task_str, variation)

            move = Mover(task, max_tries=max_tries)
            reward = 0.0
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                if step_id > 0:
                    self.obs_history = self.obs_history[::-1][::self.obs_history_augmentation_every_n][::-1][-num_history:]

                for obs in self.obs_history:
                    keypoint, keypoint_idx, gripper = self.get_keypoints_gripper_from_obs(obs)
                    keypoint_idx = keypoint_idx.to(device).reshape(1,1,*keypoint_idx.shape)
                    keypoint = keypoint.to(device).reshape(1,1,*keypoint.shape)
                    gripper = gripper.to(device).reshape(1,1,*gripper.shape)

                    keypoint_idxs = torch.cat([keypoint_idxs, keypoint_idx], dim=1)
                    keypoints = torch.cat([keypoints, keypoint], dim=1)
                    grippers = torch.cat([grippers, gripper], dim=1)

                self.obs_history = []

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                keypoint_input = keypoints[:, -num_history:][:, :, :, :3]
                keypoint_idxs_input = keypoint_idxs[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, 0, 0, npad, 0), mode='replicate'
                )
                keypoint_input = F.pad(
                    keypoint_input, (0, 0, 0, 0, npad, 0), mode='replicate'
                )
                keypoint_idxs_input = F.pad(
                    keypoint_idxs_input, (0, 0, npad, 0), mode='replicate'
                )

                action = actioner.predict(
                    keypoint_idxs_input,
                    keypoint_input,
                    gripper_input,
                )

                if verbose:
                    print(f"Step {step_id}")

                terminate = True

                # Update the observation based on the predicted action
                try:
                    print("Plan with RRT")
                    collision_checking = self._collision_checking(task_str, step_id)
                    obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0
                    #break

            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )

        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        return success_rate, valid, num_valid_demos

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        # collision_checking = True
        collision_checking = False
        # if task_str == 'close_door':
        #     collision_checking = True
        # if task_str == 'open_fridge' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'open_oven' and step_id == 3:
        #     collision_checking = True
        # if task_str == 'hang_frame_on_hanger' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'take_frame_off_hanger' and step_id == 0:
        #     for i in range(300):
        #         self.env._scene.step()
        #     collision_checking = True
        # if task_str == 'put_books_on_bookshelf' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'slide_cabinet_open_and_place_cups' and step_id == 0:
        #     collision_checking = True
        return collision_checking

    def verify_demos(
        self,
        task_str: str,
        variation: int,
        num_demos: int,
        max_tries: int = 1,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        success_rate = 0.0
        invalid_demos = 0

        for demo_id in range(num_demos):
            if verbose:
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
            except:
                print(f"Invalid demo {demo_id} for {task_str} variation {variation}")
                print()
                traceback.print_exc()
                invalid_demos += 1

            task.reset_to_demo(demo)

            gt_keyframe_actions = []
            for f in keypoint_discovery(demo):
                obs = demo[f]
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                gt_keyframe_actions.append(action)

            move = Mover(task, max_tries=max_tries)

            for step_id, action in enumerate(gt_keyframe_actions):
                if verbose:
                    print(f"Step {step_id}")

                try:
                    obs, reward, terminate, step_images = move(action)
                    if reward == 1:
                        success_rate += 1 / num_demos
                        break
                    if terminate and verbose:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_type, demo, success_rate, e)
                    reward = 0
                    break

            if verbose:
                print(f"Finished demo {demo_id}, SR: {success_rate}")

        # Compensate for failed demos
        if (num_demos - invalid_demos) == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = success_rate * num_demos / (num_demos - invalid_demos)
            valid = True

        self.env.shutdown()
        return success_rate, valid, invalid_demos

    def create_obs_config(
        self
    ):
        """
        Set up observation config for RLBench environment.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        obs_config = ObservationConfig(
            front_camera=unused_cams,
            left_shoulder_camera=unused_cams,
            right_shoulder_camera=unused_cams,
            wrist_camera=unused_cams,
            overhead_camera=unused_cams,
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


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc
    return torch.cat(obs, dim=0)



import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

def test():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_diffusion_unet_lowdim_relative_workspace")

    OmegaConf.resolve(cfg)

    env = RLBenchLowDimEnv(
        data_path="/home/felix/Workspace/diffusion_policy_felix/data/train",
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=2
    )

    # env.verify_demos("open_drawer", 0, 1, max_tries=1, verbose=True)

    task_str = "open_drawer"
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task_variations = task.variation_count()
    task.set_variation(0)

    policy : DiffusionUnetLowDimRelativePolicy = hydra.utils.instantiate(cfg.policy)
    checkpoint_path = "/home/felix/Workspace/diffusion_policy_felix/data/outputs/2024.03.20/11.54.51_diffusion_unet_lowdim_relative_policy_open_drawer/checkpoints/latest.ckpt"
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    
    actioner = Actioner(
        policy=policy,
        instructions={"open_drawer": [[torch.rand(10)],[torch.rand(10)],[torch.rand(10)]]},
        action_dim=8,
    )

    env._evaluate_task_on_one_variation(
        task_str="open_drawer",
        task=task,
        max_steps=30,
        variation=0,
        num_demos=1,
        actioner=actioner,
        max_tries=1,
        verbose=True,
        dense_interpolation=False,
        interpolation_length=50,
        num_history=policy.n_obs_steps,
    )
    # env.env.launch()


if __name__ == "__main__":
    test()
    print("Done!")