import os
import glob

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from diffusion_policy.env.rlbench.rlbench_utils import Mover, Actioner, task_file_to_task_class, keypoint_discovery, transform, get_object_pose_indices_from_task
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from diffusion_policy.common.launch_utils import object_poses_to_keypoints
from scipy.spatial.transform import Rotation as R
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.rlbench.rlbench_utils import CircleCameraMotion

class RLBenchLowDimEnv:

    def __init__(
        self,
        data_path,
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=10,
        output_dir=None
    ):

        # setup required inputs
        self.data_path = data_path
        self.output_dir = output_dir

        # setup RLBench environments
        self.obs_config = self.create_obs_config()


        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.obs_history = []
        self.obs_history_augmentation_every_n = obs_history_augmentation_every_n
        self.action_mode.arm_action_mode.set_callable_each_step(self.save_observation)
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )

        self._rgbs = []
        self._recording = False
        self._cam = None

    def launch(self):
        if self.env._pyrep is not None:
            if self._cam is None:
                cam = VisionSensor.create([128, 128])
                cam.set_pose(Dummy('cam_cinematic_placeholder').get_pose())
                cam.set_parent(Dummy('cam_cinematic_placeholder'))
                self._cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
                self._cam = cam
            return
        self.env.launch()

    def shutdown(self):
        if self.env._pyrep is None:
            return
        self.env.shutdown()
        self._cam_motion = None
        self._cam = None

    def save_observation(self, obs):
        """
        Save the observation to the history
        :param obs: an Observation from the env
        """
        self.obs_history.append(obs)
        if self._recording:
            self._rgbs.append((self._cam.capture_rgb() * 255.).astype(np.uint8))
            self._cam_motion.step()

    def start_recording(self):
        self._recording = True
            
    def stop_recording(self):
        self._recording = False

    def save_video(self):
        if len(self._rgbs) == 0:
            return None
        path = os.path.join(self.output_dir, "media", wv.util.generate_id() + ".mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import imageio.v2 as iio
        writer = iio.get_writer(path, fps=30, format='FFMPEG', mode='I')
        for image in self._rgbs:
            writer.append_data(image)
        writer.close()
        self._rgbs = []
        self._cam_motion.restore_pose()
        return path
    
    def get_rgbs(self):
        if len(self._rgbs) == 0:
            return None
        rgbs = np.array(self._rgbs)
        rgbs = rgbs.transpose(0, 3, 1, 2)
        self._rgbs = []
        self._cam_motion.restore_pose() 
        return np.array(rgbs)

    def get_keypoints_gripper_from_obs(self, obs, object_pose_indices):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        object_poses = np.array([obs.task_low_dim_state[i:i+7] for i in object_pose_indices])
        pcd = torch.from_numpy(object_poses_to_keypoints(object_poses))
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
        num_history=1,
    ):
        self.launch()
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
                    max_rtt_tries=max_tries,
                    verbose=verbose,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates


    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_rtt_tries: int = 1,
        demo_tries: int = 1,
        verbose: bool = False,
        num_history=0,
    ):
        self.launch()
        demos = []
        num_valid_demos = 0
        for i in range(num_demos):
            try:
                demo = self.get_demo(task_str, variation, episode_index=i)[0]
                demos.append(demo)
                num_valid_demos += 1
            except:
                print(f"Invalid demo {i} for {task_str} variation {variation}")
                print()
                traceback.print_exc()

        log_data = self._evaluate_task_on_demos(
            task=task,
            task_str=task_str,
            demos=demos,
            max_steps=max_steps,
            actioner=actioner,
            max_rtt_tries=max_rtt_tries,
            demo_tries=demo_tries,
            verbose=verbose,
            num_history=num_history,
        )

        self.shutdown()

        successful_demos = log_data["success_rate"]


        if num_valid_demos == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = successful_demos / (num_valid_demos * demo_tries)
            valid = True

        return success_rate, valid, num_valid_demos

    @torch.no_grad()

    def record_demos(self,
                    task: TaskEnvironment,
                    task_str: str,
                    demos: List[Demo],  
                    max_steps: int,
                    actioner: Actioner,
                    max_rtt_tries: int = 1,
                    verbose: bool = False,
                    num_history=0):
        video_paths = []
        self.start_recording()
        for demo in demos:
            self._cam_motion.restore_pose()
            self._evaluate_task_on_demos(
                task=task,
                task_str=task_str,
                demos=[demo],
                max_steps=max_steps,
                actioner=actioner,
                max_rtt_tries=max_rtt_tries,
                demo_tries=1,
                verbose=verbose,
                num_history=num_history,
            )            
            video_path = self.save_video()
            if video_path is not None:
                video_paths.append(video_path)
        self.stop_recording()
        return video_paths

    @torch.no_grad()
    def _evaluate_task_on_demos(self,       
                                task: TaskEnvironment,
                                task_str: str,
                                demos: List[Demo],  
                                max_steps: int,
                                actioner: Actioner,
                                max_rtt_tries: int = 1,
                                demo_tries: int = 1,
                                n_visualize: int = 0,
                                verbose: bool = False,
                                num_history=0):
        self.launch()
        device = actioner.device
        successful_demos = 0
        total_reward = 0
        video_paths = []

        for demo_id, demo in enumerate(demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            for i in range(demo_tries):
                if demo_id < n_visualize and i == 0:
                    self.start_recording()

                keypoint_idxs = torch.Tensor([]).to(device)
                keypoints = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # descriptions, obs = task.reset()
                descriptions, obs = task.reset_to_demo(demo)
                object_pose_indices = get_object_pose_indices_from_task(task._task)
                self.obs_history.append(obs)

                actioner.load_episode(task_str, demo.variation_number)
                # actioner.load_demo(demo) # TODO: remove this line

                move = Mover(task, max_tries=max_rtt_tries)
                reward = 0.0
                max_reward = 0.0

                for step_id in range(max_steps):

                    # Fetch the current observation, and predict one action
                    if step_id > 0:
                        self.obs_history = self.obs_history[::-1][::self.obs_history_augmentation_every_n][::-1][-num_history:]

                    for obs in self.obs_history:
                        keypoint, keypoint_idx, gripper = self.get_keypoints_gripper_from_obs(obs, object_pose_indices)
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
                        if verbose:
                            print("Plan with RRT")
                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                        max_reward = max(max_reward, reward)

                        if reward == 1:
                            successful_demos += 1
                            break

                        if terminate:
                            print("The episode has terminated!")

                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(task_str, demo, step_id, successful_demos, e)
                        reward = 0
                        #break

                if demo_id < n_visualize and i == 0:
                    self.stop_recording()
                    video_path = self.save_video()
                    video_paths.append(video_path)

                total_reward += max_reward
                if reward == 0:
                    step_id += 1

                print(
                    task_str,
                    "Variation",
                    demo.variation_number,
                    "Demo",
                    demo_id,
                    "Reward",
                    f"{reward:.2f}",
                    "max_reward",
                    f"{max_reward:.2f}",
                    f"SR: {successful_demos / ((demo_id+1) * demo_tries)}",
                    f"Total reward: {total_reward:.2f}/{(demo_id+1) * demo_tries}"                )

        self.shutdown()
        log_data = {
            "success_rate": successful_demos / (len(demos) * demo_tries),
            "total_reward": total_reward,
            "video_paths": video_paths,
        }
        return log_data
    
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
        demo_consistency_tries: int = 10,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        demo_success_rates = []
        demo_valid = np.array([], dtype=bool)

        with tqdm(range(num_demos), desc=f"Validated demos", 
                leave=False) as tdemos:
            for demo_id in tdemos:
                if verbose:
                    print(f"Starting demo {demo_id}")

                try:
                    demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                except:
                    print(f"Invalid demo {demo_id} for {task_str} variation {variation}")
                    print()
                    traceback.print_exc()
                    demo_valid = np.append(demo_valid, False)
                    continue

                successful_ties = 0

                for i in range(demo_consistency_tries):
                    task.reset_to_demo(demo)

                    gt_keyframe_actions = []
                    gt_obs = []
                    for f in keypoint_discovery(demo):
                        obs = demo[f]
                        gt_obs.append(obs)
                        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                        gt_keyframe_actions.append(action)

                    move = Mover(task, max_tries=max_tries)

                    for step_id, action in enumerate(gt_keyframe_actions):
                        if verbose:
                            print(f"Step {step_id}")

                        try:
                            obs, reward, terminate, step_images = move(action)
                            if reward == 1:
                                successful_ties += 1
                                break
                            if terminate and verbose:
                                print("The episode has terminated!")

                        except (IKError, ConfigurationPathError, InvalidActionError) as e:
                            print(task_type, demo, success_rate, e)
                            reward = 0
                            break
                                    
                
                if successful_ties == demo_consistency_tries:
                    demo_valid = np.append(demo_valid, True)
                else:
                    demo_valid = np.append(demo_valid, False)
                
                if verbose:
                    print(f"Finished demo {demo_id}, SR: {np.count_nonzero(demo_valid) / num_demos}")
                demo_success_rates.append(successful_ties)
                
                tdemos.set_postfix(SR=np.count_nonzero(demo_valid) / num_demos, refresh=False)
        # Compensate for failed demos
        if (num_demos - np.count_nonzero(~demo_valid)) == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = np.count_nonzero(demo_valid)/num_demos
            valid = True

        self.env.shutdown()
        return success_rate, demo_valid, demo_success_rates

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


import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

def test():
    from diffusion_policy.policy.diffusion_unet_lowdim_relative_policy import DiffusionUnetLowDimRelativePolicy
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_flow_matching_unet_lowdim_workspace")

    OmegaConf.resolve(cfg)



    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/peract"
    env = RLBenchLowDimEnv(
        data_path=data_path,
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=2,
        output_dir="/home/felix/Workspace/diffusion_policy_felix/data/dummy"
    )

    n_demos = 1
    task_str = "turn_tap"
    variation = 0
    # sr, demo_valid, success_rates = env.verify_demos(task_str, 0, n_demos, max_tries=10, demo_consistency_tries=1, verbose=True)

    # print("Success rate: ", sr)
    # print("Valid demos: ", valid_demos)
    # print("Invalid demos: ", invalid_demos)
    # print("Success rates: ", success_rates)
    # return

    
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task_variations = task.variation_count()
    task.set_variation(0)

    policy : DiffusionUnetLowDimRelativePolicy = hydra.utils.instantiate(cfg.policy)
    checkpoint_path = "/home/felix/Workspace/diffusion_policy_felix/data/outputs/2024.04.12/11.31.41_flow_matching_unet_lowdim_policy_turn_tap/checkpoints/epoch=93750-val_loss=0.006.ckpt"
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    
    actioner = Actioner(
        policy=policy,
        instructions={task_str: [[torch.rand(10)],[torch.rand(10)],[torch.rand(10)]]},
        action_dim=8,
    )

    class ReplayActionerActioner(Actioner):
        def __init__(self, policy, instructions, action_dim, task_str, variation):
            super().__init__(policy=policy, instructions=instructions, action_dim=action_dim)
            self._actions, self._keypoints = self.get_action_from_demo(env.get_demo(task_str, variation, episode_index=0)[0])
            self.idx = 0
        def predict(self, keypoint_idxs, keypoints, gripper):
            if self.idx % 2 == 0:
                self.idx += 1
                return super().predict(keypoint_idxs, keypoints, gripper)
            else:
                self.idx += 1
                return self._actions.pop(0).squeeze().numpy()
        
        # def load_episode(self, task_str, variation):
        #     pass

        def load_demo(self, demo):
            self._actions, self._keypoints = self.get_action_from_demo(demo)

        def get_action_from_demo(self, demo):
            """
            Fetch the desired state and action based on the provided demo.
                :param demo: fetch each demo and save key-point observations
                :return: a list of obs and action
            """
            key_frame = keypoint_discovery(demo)

            action_ls = []
            trajectory_ls = []
            keypoint_ls = []
            keypoint_ls.append(object_poses_to_keypoints(demo[0].misc['object_poses']))
            for i in range(len(key_frame)):
                obs = demo[key_frame[i]]
                action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                action = torch.from_numpy(action_np)
                action_ls.append(action.unsqueeze(0))

                keypoints = object_poses_to_keypoints(obs.misc['object_poses'])
                keypoint_ls.append(torch.from_numpy(keypoints).unsqueeze(0))
            return action_ls, keypoint_ls
        
    replay_actioner = ReplayActionerActioner(policy, instructions={task_str: [[torch.rand(10)],[torch.rand(10)],[torch.rand(10)]]},
        action_dim=8, task_str=task_str, variation=0)

    env._evaluate_task_on_one_variation(
        task_str=task_str,
        task=task,
        max_steps=3,
        variation=0,
        num_demos=1,
        actioner=actioner,
        max_rtt_tries=10,
        demo_tries=1,
        verbose=True,
        num_history=policy.n_obs_steps,
    )
    env.save_video()
    env.env.shutdown()


if __name__ == "__main__":
    test()
    print("Done!")