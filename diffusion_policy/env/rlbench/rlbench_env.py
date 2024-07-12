import os
import glob

from pytorch3d.transforms import quaternion_to_matrix
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops
from typing import List
from diffusion_policy.common.pytorch_util import dict_apply, compare_dicts
from rlbench.environment import Environment
from rlbench.demo import Demo
from rlbench.backend.observation import Observation
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from diffusion_policy.env.rlbench.rlbench_utils import Mover, Actioner, task_file_to_task_class, get_actions_from_demo
from pyrep.errors import IKError, ConfigurationPathError
from scipy.spatial.transform import Rotation as R
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.rlbench.rlbench_utils import CircleCameraMotion
from diffusion_policy.common.rlbench_util import extract_obs, create_obs_config, _keypoint_discovery, create_obs_state_plot, CAMERAS
from diffusion_policy.common.pytorch_util import print_dict
from diffusion_policy.common.visualization_se3 import visualize_frames, visualize_poses_and_actions
from diffusion_policy.model.common.so3_util import quaternion_to_matrix


def visualize(obs, action=None):
    state_rotation = [v for k, v in obs.items() if 'rot' in k]
    state_translation = [v for k, v in obs.items() if 'pos' in k]

    state_rotation = torch.cat(state_rotation, dim=1).flatten(0, 1)
    state_translation = torch.cat(state_translation, dim=1).flatten(0, 1)

    if action is not None:
        action_rotation = action['act_r'].flatten(0, 1)
        action_translation = action['act_p'].flatten(0, 1)
        visualize_poses_and_actions(state_rotation, state_translation, action_rotation, action_translation)
    else:
        visualize_frames(state_rotation, state_translation)


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

def create_axis(color = [1, 0, 0], name = 'x', size=[0.01, 0.01, 0.1], parent=None):
    axis = Shape.create(type=PrimitiveShape.CYLINDER, 
                                        size=size,
                                        static=True,
                                        respondable=False,
                                        color=color,
                                        renderable=True,
                                        orientation=[0, 0, 0],)
    
    if parent is not None:
        if name == 'x':
            axis.set_position([size[2] / 2, 0, 0], relative_to=parent)
            axis.set_orientation([0, np.pi / 2, 0], relative_to=parent)
        elif name == 'y':
            axis.set_position([0, size[2] / 2, 0], relative_to=parent)
            axis.set_orientation([-np.pi / 2, 0, 0], relative_to=parent)
        elif name == 'z':
            axis.set_position([0, 0, size[2] / 2], relative_to=parent)
            axis.set_orientation([0, 0, 0], relative_to=parent)
        axis.set_parent(parent)
        
    axis.set_name(name)
    return axis


def create_coordinate_frame(size=[0.01, 0.01, 0.1]):
    # Create a coordinate frame
    sphere = Shape.create(type=PrimitiveShape.SPHERE,
                            size=[0.01, 0.01, 0.01],
                            static=True,
                            respondable=False,
                            color=[1, 1, 1],
                            renderable=True,
                            orientation=[0, 0, 0])
    create_axis(color=[1, 0, 0], name='x', size=size, parent=sphere)
    create_axis(color=[0, 1, 0], name='y', size=size, parent=sphere)
    create_axis(color=[0, 0, 1], name='z', size=size, parent=sphere)
    return sphere

def create_keypoint(pos, size=[0.002, 0.002, 0.02]):
    keypoint = Dummy.create()
    keypoint.set_position(pos)
    z = Shape.create(type=PrimitiveShape.CYLINDER, 
                                        size=size,
                                        static=True,
                                        respondable=False,
                                        color=[0, 0, 1],
                                        renderable=True,
                                        orientation=[0, 0, 0],)
    z.set_position(pos)
    z.set_parent(keypoint)
    x = Shape.create(type=PrimitiveShape.CYLINDER,
                                        size=size,
                                        static=True,
                                        respondable=False,
                                        color=[1, 0, 0],
                                        renderable=True,
                                        orientation=[0, 0, 0],)
    x.set_position(pos)
    x.set_orientation([0, np.pi / 2, 0])
    x.set_parent(z)
    y = Shape.create(type=PrimitiveShape.CYLINDER,
                                        size=size,
                                        static=True,
                                        respondable=False,
                                        color=[0, 1, 0],
                                        renderable=True,
                                        orientation=[0, 0, 0],)
    y.set_position(pos)
    y.set_orientation([-np.pi / 2, 0, 0])
    y.set_parent(z)
    return keypoint

class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_mask=False,
        apply_low_dim_pcd=False,
        apply_pose=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False,
        obs_history_from_planner=True,
        obs_history_augmentation_every_n=10,
        n_obs_steps=1,
        n_action_steps=1,
        render_image_size=[128, 128],
        adaptor=None,
        visualize_ghost_objects=False,
        **kwargs
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_mask = apply_mask
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.apply_low_dim_pcd = apply_low_dim_pcd
        self.apply_poses = apply_pose
        self.adaptor = adaptor

        # setup RLBench environments
        self.obs_config = create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_mask, apply_cameras, **kwargs
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.obs_history = []
        self.obs_history_from_planner = obs_history_from_planner
        self.obs_history_augmentation_every_n = obs_history_augmentation_every_n
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_mode.arm_action_mode.set_callable_each_step(self.rrt_callable_each_step)
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size
        self._render_img_size = render_image_size
        self._rgbs = []
        self._recording = False
        self._cam = None
        self._gripper_dummmies = None
        self._observation_dummies = None
        self._keypoints = None
        self.visualize_ghost_objects = visualize_ghost_objects
        
    
    def create_gripper_dummies(self, num_grippers) -> Dummy:
        
        gripper_dummies = []
        for _ in range(num_grippers):
            gripper_shapes = [shape.copy() for shape in self.env._robot.gripper.get_visuals()]
            gripper_dummy = Dummy.create()

            gripper_shapes[0].set_parent(gripper_dummy)
            gripper_shapes[0].set_transparency(0.5)
            gripper_shapes[1].set_parent(gripper_shapes[0])
            gripper_shapes[1].set_transparency(0.5)
            gripper_shapes[2].set_parent(gripper_shapes[0])
            gripper_shapes[2].set_transparency(0.5)

            gripper_shapes[0].set_position([0, 0, -0.09], relative_to=gripper_dummy)
            gripper_shapes[0].set_orientation([np.pi/2, 0, 0], relative_to=gripper_dummy)

            gripper_dummies.append(gripper_dummy)

        return gripper_dummies
    
    def set_gripper_dummies(self, trajectory):
        for gripper_dummy, gripper_pose in zip(self._gripper_dummmies, trajectory):
            gripper_dummy.set_pose(gripper_pose)
            self.set_gripper_renderable(gripper_dummy, True)

    def set_gripper_renderable(self, gripper_dummy, renderable):
        for child in gripper_dummy.get_objects_in_tree():
            child.set_renderable(renderable)

    def launch(self):
        if self.env._pyrep is None:
            self.env.launch()
        if self._cam is None:
            cam = VisionSensor.create(self._render_img_size)
            cam.set_pose(Dummy('cam_cinematic_placeholder').get_pose())
            cam.set_parent(Dummy('cam_cinematic_placeholder'))
            self._cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005, init_rotation=np.deg2rad(0))
            self._cam = cam
        if self.visualize_ghost_objects:
            if self._gripper_dummmies is None:
                self._gripper_dummmies = self.create_gripper_dummies(self.n_action_steps)
            if self._observation_dummies is None:
                self._observation_dummies = self.create_gripper_dummies(self.n_obs_steps) 

    def shutdown(self):
        self.env.shutdown()
        self._cam_motion = None
        self._cam = None
        self._gripper_dummmies = None
        self._observation_dummies = None
        self._keypoints = None

    def reset(self):
        self._rgbs = []
        self._cam_motion.restore_pose()
        self._keypoints = None
        self._recording = False


    def create_keypoints(self, positions):
        keypoints = []
        for pos in positions:
            keypoints.append(create_keypoint(pos))
        return keypoints

    def update_keypoints(self, positions):
        if self._keypoints is None:
            self._keypoints = self.create_keypoints(positions)
        else:
            for keypoint, pos in zip(self._keypoints, positions):
                keypoint.set_position(pos)
        
    def rrt_callable_each_step(self, obs):
        """
        Save the observation to the history
        :param obs: an Observation from the env
        """
        if self.obs_history_from_planner:
            self.store_obs(obs)
        if self._recording:
            self.record_frame(obs)

    def record_frame(self, obs):
        if self.visualize_ghost_objects:
            if "low_dim_pcd" in obs.misc:
                self.update_keypoints(obs.misc["low_dim_pcd"])
        rgb = (self._cam.capture_rgb() * 255.).astype(np.uint8)
        self._rgbs.append(rgb)
        self._cam_motion.step()
        return rgb

    def draw_observation_history(self, n_frames=30):
        for obs, dummy in zip(self.get_obs_history(), self._observation_dummies):
            dummy.set_pose(obs.gripper_pose)
            self.set_gripper_renderable(dummy, True)

        for _ in range(n_frames):
            self.env._scene.step()
            self.record_frame(self.obs_history[-1])

        for dummy in self._observation_dummies:
            self.set_gripper_renderable(dummy, False)
    
    def reset_obs_history(self):
        self.obs_history = []

    def store_obs(self, obs):
        self.obs_history.append(obs)
        if self.obs_history_from_planner:
            self.obs_history = self.obs_history[::-1][:self.obs_history_augmentation_every_n * self.n_obs_steps][::-1]
        else:
            self.obs_history = self.obs_history[-1:self.n_obs_steps]
            
    def get_obs_history(self):
        if self.obs_history_from_planner:
            observations = self.obs_history[::-1][::self.obs_history_augmentation_every_n][::-1][-self.n_obs_steps:]
        else:
            observations = self.obs_history[-1:]
        return observations

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

    def get_obs_dict(self, obs):
        """
        Fetch the desired state based on the provided demo.
            :param obs: incoming obs
            :return: required observation
        """
        obs_dict = dict()

        if self.apply_rgb:
            obs_dict["rgb"] = []

        if self.apply_depth:
            obs_dict["depth"] = []

        if self.apply_pc:
            obs_dict["pcd"] = []
        
        if self.apply_mask:
            obs_dict["mask"] = []

        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                obs_dict["rgb"] += [torch.from_numpy(rgb)]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                obs_dict["depth"] += [torch.from_numpy(depth)]

            if self.apply_pc:
                pcd = getattr(obs, "{}_point_cloud".format(cam))
                obs_dict["pcd"] += [torch.from_numpy(pcd)]

            if self.apply_mask:
                mask = getattr(obs, "{}_mask".format(cam))
                obs_dict["mask"] += [torch.from_numpy(mask)]

        if self.apply_rgb:
            rgb = torch.stack(obs_dict["rgb"])
            rgb = rgb.permute(0, 3, 1, 2)
            rgb = rgb / 255.0
            rgb = rgb.float()
            obs_dict["rgb"] = rgb

        if self.apply_depth:
            depth = torch.stack(obs_dict["depth"])
            depth = depth.float()
            obs_dict["depth"] = depth  

        if self.apply_pc:
            pcd = torch.stack(obs_dict["pcd"])
            pcd = pcd.permute(0, 3, 1, 2)
            obs_dict["pcd"] = pcd

        if self.apply_mask:
            mask_ids = torch.stack(obs_dict["mask"]).int()
            obs_dict["mask"] = mask_ids.unsqueeze(1)

        if self.apply_low_dim_pcd:
            low_dim_pcd = obs.misc["low_dim_pcd"]
            obs_dict["low_dim_pcd"] = torch.from_numpy(low_dim_pcd).float()

        if self.apply_poses:
            keypoint_poses = obs.misc["low_dim_poses"]
            obs_dict["keypoint_poses"] = torch.from_numpy(keypoint_poses).float()

        gripper = torch.from_numpy(obs.gripper_matrix).float()
        low_dim_state = torch.from_numpy(
            np.array([obs.gripper_open, *obs.gripper_joint_positions])).float()

        obs_dict["gripper"] = gripper
        obs_dict["low_dim_state"] = low_dim_state

        return obs_dict

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
    
    def verify_demo(            
            self,
            demo: Demo,
            task: TaskEnvironment,
            max_rrt_tries: int = 1,
            demo_consistency_tries: int = 10,
            verbose: bool = False,
    ):
        return self._verify_demo(
            demo, task, max_rrt_tries, demo_consistency_tries, verbose
        ) == demo_consistency_tries

    
    def _verify_demo(
            self,
            demo: Demo,
            task: TaskEnvironment,
            max_rrt_tries: int = 1,
            demo_consistency_tries: int = 10,
            verbose: bool = False,
    ):
        self.launch()

        successful_tries = 0

        for i in range(demo_consistency_tries):
            task.reset_to_demo(demo)
            gt_keyframe_actions = []
            gt_obs = []
            for f in _keypoint_discovery(demo):
                obs = demo[f]
                gt_obs.append(obs)
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                gt_keyframe_actions.append(action)

            move = Mover(task, max_tries=max_rrt_tries)

            for step_id, action in enumerate(gt_keyframe_actions):
                if verbose:
                    print(f"Step {step_id}")

                try:
                    obs, reward, terminate, step_images = move(action)
                    if reward == 1:
                        successful_tries += 1
                        break
                    if terminate and verbose:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    reward = 0
                    break
                            
        return successful_tries

    def verify_demos(
        self,
        task_str: str,
        variation: int,
        num_demos: int,
        max_rrt_tries: int = 1,
        demo_consistency_tries: int = 10,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.launch()
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

                successful_ties = self._verify_demo(
                    demo, task, max_rrt_tries, demo_consistency_tries, verbose
                )
                
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
