import os
import glob

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops
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

# computes pixel location of gripper in the image
def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v

class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False,
        obs_history_augmentation_every_n=10,
        render_img_size=[128, 128]
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )


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
        self.image_size = image_size
        self._render_img_size = render_img_size
        self._rgbs = []
        self._recording = False
        self._cam = None

    def launch(self):
        if self.env._pyrep is not None:
            if self._cam is None:
                cam = VisionSensor.create(self._render_img_size)
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
    
    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, _ = self.get_obs_action(obs)
        gripper = torch.from_numpy(obs.gripper_matrix)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

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
        rgbs_ls = []

        for demo_id, demo in enumerate(demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            for i in range(demo_tries):
                if demo_id < n_visualize and i == 0:
                    self.start_recording()

                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                # descriptions, obs = task.reset()
                descriptions, obs = task.reset_to_demo(demo)
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
                        rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                        rgb = rgb.to(device)
                        pcd = pcd.to(device)
                        gripper = gripper.to(device)

                        rgbs = torch.cat([rgbs, rgb], dim=0)
                        pcds = torch.cat([pcds, pcd], dim=0)
                        grippers = torch.cat([grippers, gripper], dim=0)

                    self.obs_history = []

                    # Prepare proprioception history
                    gripper_input = grippers[-num_history:].unsqueeze(0)
                    rgbs_input = rgbs[-num_history:,:,:3].unsqueeze(0) # only rgb channels, remove attns
                    pcds_input = pcds[-num_history:].unsqueeze(0)
                    npad = num_history - gripper_input.shape[1]
                    b, _, n, c, h, w = rgbs_input.shape
                    gripper_input = F.pad(
                        gripper_input, (0, 0, npad, 0), mode='replicate'
                    )
                    rgbs_input = F.pad(
                            rgbs_input.reshape(rgbs_input.shape[:2] + (-1,)),
                            (0, 0, npad, 0), mode='replicate'
                    ).view(b, -1, n, c, h, w)
                    pcds_input = F.pad(
                        pcds_input.reshape(pcds_input.shape[:2] + (-1,)),
                        (0, 0, npad, 0), mode='replicate'
                    ).view(b, -1, n, c, h, w)
                    obs = {
                        "rgbs": rgbs_input.unsqueeze(0),
                        "pcds": pcds_input.unsqueeze(0),
                        "grippers": gripper_input.unsqueeze(0),
                    }
                    action = actioner.predict(obs)

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
                    rgbs = self.get_rgbs()
                    rgbs_ls.append(rgbs)

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
            "rgbs_ls": rgbs_ls,
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
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
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
            mask=False,
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
            task_low_dim_state=False,
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

def test_verify_demos():
    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/peract"
    env = RLBenchEnv(
        data_path=data_path,
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=2,
    )
    sr, demo_valid, success_rates = env.verify_demos(
        task_str="open_drawer",
        variation=0,
        num_demos=1,
        max_rrt_tries=10,
        demo_consistency_tries=10,
        verbose=True,
    )
    print("Success rate: ", sr)
    print("Valid demos: ", np.count_nonzero(demo_valid))
    print("Invalid demos: ", np.count_nonzero(~demo_valid))
    print("Success rates: ", success_rates)
    return

def test_evaluation():
    pass

def test_replay():
    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/peract"
    env = RLBenchEnv(
        data_path=data_path,
        headless=True,
        collision_checking=False,
        obs_history_augmentation_every_n=2,
        apply_rgb=True,
        apply_pc=True
    )

    task_str = "open_drawer"
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    variation = 0
    task.set_variation(variation)

    demos = env.get_demo(task_str, variation, episode_index=0)

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
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

    class ReplayPolicy:
        def __init__(self, demo):
            self._actions, self._keypoints = get_action_from_demo(None, demo)
            self.idx = 0
            self.n_obs_steps = 2

        def predict_action(self, obs):
            return {
                "action": self._actions.pop(0).unsqueeze(0),
            }
        
        def eval(self):
            pass

        def parameters(self):
            return iter([torch.empty(0)])

    policy = ReplayPolicy(demos[0])
        
    replay_actioner = Actioner(
        policy=policy,
        instructions={task_str: [[torch.rand(10)],[torch.rand(10)],[torch.rand(10)]]},
        action_dim=8,
    )
    
    env._evaluate_task_on_demos(
        task_str=task_str,
        task=task,
        max_steps=3,
        demos=demos,
        actioner=replay_actioner,
        max_rtt_tries=10,
        demo_tries=1,
        verbose=True,
        num_history=policy.n_obs_steps,
    )
    env.env.shutdown()

if __name__ == "__main__":
    test_replay()
    # test()
    print("Done!")