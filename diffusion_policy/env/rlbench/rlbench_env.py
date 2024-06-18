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
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.demo import Demo
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from diffusion_policy.env.rlbench.rlbench_utils import Mover, Actioner, task_file_to_task_class, keypoint_discovery, transform, get_actions_from_demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from scipy.spatial.transform import Rotation as R
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.rlbench.rlbench_utils import CircleCameraMotion
from diffusion_policy.common.pytorch_util import print_dict


from diffusion_policy.common.visualization_se3 import visualize_frames, visualize_poses_and_actions
from diffusion_policy.model.common.so3_util import quaternion_to_matrix, log_map, exp_map, se3_inverse, apply_transform
def visualize(obs, action):
    state_rotation = [v for k, v in obs.items() if 'rot' in k]
    state_translation = [v for k, v in obs.items() if 'pos' in k]

    state_rotation = torch.cat(state_rotation, dim=1).flatten(0, 1)
    state_translation = torch.cat(state_translation, dim=1).flatten(0, 1)

    action_rotation = action['act_r'].flatten(0, 1)
    action_translation = action['act_p'].flatten(0, 1)
    visualize_poses_and_actions(state_rotation, state_translation, action_rotation, action_translation)


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
    z = Shape.create(type=PrimitiveShape.CYLINDER, 
                                        size=size,
                                        static=True,
                                        respondable=False,
                                        color=[0, 0, 1],
                                        renderable=True,
                                        orientation=[0, 0, 0],)
    z.set_position(pos)
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
    return x, y, z

class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        apply_low_dim_pcd=False,
        apply_pose=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False,
        obs_history_augmentation_every_n=10,
        n_obs_steps=1,
        n_action_steps=1,
        render_image_size=[128, 128],
        adaptor=None,
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.apply_low_dim_pcd = apply_low_dim_pcd
        self.apply_poses = apply_pose
        self.adaptor = adaptor

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
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_mode.arm_action_mode.set_callable_each_step(self.save_observation)
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
        self._keypoints = None
    
    def create_gripper_dummies(self) -> Dummy:
        
        gripper_dummies = []
        for _ in range(self.n_action_steps):
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
        if self._gripper_dummmies is None:
            self._gripper_dummmies = self.create_gripper_dummies() 

    def shutdown(self):
        self.env.shutdown()
        self._cam_motion = None
        self._cam = None

    def add_keypoints(self, positions):
        if self._keypoints is not None:
            for keypoint in self._keypoints:
                keypoint.remove()
        keypoints = []
        for pos in positions:
            keypoints.extend(create_keypoint(pos))
        self._keypoints = keypoints


    def save_observation(self, obs):
        """
        Save the observation to the history
        :param obs: an Observation from the env
        """
        self.obs_history.append(obs)
        if self._recording:
            self.add_keypoints(obs.misc["low_dim_pcd"])
            rgb = (self._cam.capture_rgb() * 255.).astype(np.uint8)
            self._rgbs.append(rgb)
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
        
        if self.apply_low_dim_pcd:
            low_dim_pcd = obs.misc["low_dim_pcd"]
            state_dict["low_dim_pcd"] = low_dim_pcd

        if self.apply_poses:
            keypoint_poses = obs.misc["low_dim_poses"]
            state_dict["keypoint_poses"] = keypoint_poses

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_low_dim_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, _ = self.get_obs_action(obs)
        out_dict = dict()
        gripper = torch.from_numpy(obs.gripper_matrix)
        gripper = gripper.unsqueeze(0)  # 1, D
        out_dict['gripper'] = gripper
        low_dim_state = np.array([obs.gripper_open, *obs.gripper_joint_positions])
        low_dim_state = torch.from_numpy(low_dim_state).unsqueeze(0) # 1, D
        out_dict['low_dim_state'] = low_dim_state
        if self.apply_low_dim_pcd:
            out_dict['low_dim_pcd'] = state_dict.pop('low_dim_pcd').unsqueeze(0)  # 1, N, 3
        if self.apply_poses:
            out_dict['keypoint_poses'] = state_dict.pop('keypoint_poses').unsqueeze(0)  # 1, N, 4, 4

        if self.apply_rgb and self.apply_pc:
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

            attns = torch.Tensor([])
            for cam in self.apply_cameras:
                u, v = obs_to_attn(obs, cam)
                attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
                if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                    attn[0, 0, 0, v, u] = 1
                attns = torch.cat([attns, attn], 1)
            rgb = torch.cat([rgb, attns], 2)

            out_dict['rgb'] = rgb
            out_dict['pcd'] = pcd

        return out_dict       

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

        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                obs_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                obs_dict["depth"] += [depth]

            if self.apply_pc:
                pcd = getattr(obs, "{}_point_cloud".format(cam))
                obs_dict["pcd"] += [pcd]

        if self.apply_rgb:
            rgb = torch.stack(obs_dict["rgb"])
            rgb = rgb.permute(0, 3, 1, 2)
            rgb = rgb / 255.0
            rgb = rgb * 2 - 1
            rgb = rgb.float()
            obs_dict["rgb"] = rgb

        if self.apply_depth:
            depth = torch.stack(obs_dict["depth"])
            depth = depth.float()
            obs_dict["depth"] = depth  

        if self.apply_pc:
            pcd = torch.stack(obs_dict["pc"])
            pcd = pcd.permute(0, 2, 1)
            obs_dict["pcd"] = pcd

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
        rotation = quaternion_to_matrix(torch.tensor([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])).numpy()
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
                                verbose: bool = False):
        self.launch()
        device = actioner.device
        dtype = actioner.dtype
        successful_demos = 0
        total_reward = 0
        rgbs_ls = []
        n_obs_steps = self.n_obs_steps

        for demo_id, demo in enumerate(demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

            for i in range(demo_tries):
                if demo_id < n_visualize and i == 0:
                    self.start_recording()

                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                low_dim_pcds = torch.Tensor([]).to(device)
                keypoint_poses = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)
                low_dim_states = torch.Tensor([]).to(device)

                # descriptions, obs = task.reset()
                descriptions, observation = task.reset_to_demo(demo)
                self.obs_history.append(observation)

                actioner.load_episode(task_str, demo.variation_number)

                move = Mover(task, max_tries=max_rtt_tries)
                reward = 0.0
                max_reward = 0.0

                for step_id in range(max_steps):

                    # Fetch the current observation, and predict one action
                    if step_id > 0:
                        self.obs_history = self.obs_history[::-1][::self.obs_history_augmentation_every_n][::-1][-n_obs_steps:]

                    for observation in self.obs_history:
                        obs_dict = self.get_obs_dict(observation)
                        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(device))

                        if self.apply_rgb:
                            rgb = obs_dict['rgb'].to(device)
                            rgbs = torch.cat([rgbs, rgb], dim=0)
                        if self.apply_pc:
                            pcd = obs_dict['pcd'].to(device)
                            pcds = torch.cat([pcds, pcd], dim=0)
                        if self.apply_low_dim_pcd:
                            low_dim_pcd = obs_dict['low_dim_pcd'].to(device)
                            low_dim_pcds = torch.cat([low_dim_pcds, low_dim_pcd], dim=0)
                        if self.apply_poses:
                            keypoint_pose = obs_dict['keypoint_poses'].to(device)
                            keypoint_poses = torch.cat([keypoint_poses, keypoint_pose], dim=0)

                        gripper = obs_dict['gripper'].to(device)
                        grippers = torch.cat([grippers, gripper], dim=0)
                        low_dim_state = obs_dict['low_dim_state'].to(device)
                        low_dim_states = torch.cat([low_dim_states, low_dim_state], dim=0)
                    

                    # Prepare proprioception history
                    gripper_input = grippers[-n_obs_steps:].unsqueeze(0)

                    npad = n_obs_steps - gripper_input.shape[1]
                    gripper_input = F.pad(
                        gripper_input, (0, 0, 0, 0, npad, 0), mode='replicate'
                    )
                    low_dim_state_input = low_dim_states[-n_obs_steps:].unsqueeze(0)
                    low_dim_state_input = F.pad(
                        low_dim_state_input, (0, 0, npad, 0), mode='replicate'
                    )   
                    if self.apply_rgb:
                        rgbs_input = rgbs[-n_obs_steps:,:,:3].unsqueeze(0) # only rgb channels, remove attns
                        b, _, n, c, h, w = rgbs_input.shape
                        rgbs_input = F.pad(
                                rgbs_input.reshape(rgbs_input.shape[:2] + (-1,)),
                                (0, 0, npad, 0), mode='replicate'
                        ).view(b, -1, n, c, h, w)

                    if self.apply_pc:
                        pcds_input = pcds[-n_obs_steps:].unsqueeze(0)
                        b, _, n, c, h, w = pcds_input.shape
                        pcds_input = F.pad(
                            pcds_input.reshape(pcds_input.shape[:2] + (-1,)),
                            (0, 0, npad, 0), mode='replicate'
                        ).view(b, -1, n, c, h, w)

                    if self.apply_low_dim_pcd:
                        low_dim_pcds_input = low_dim_pcds[-n_obs_steps:].unsqueeze(0)
                        low_dim_pcds_input = F.pad(
                            low_dim_pcds_input,
                            (0, 0, 0, 0, npad, 0), mode='replicate'
                        )

                    if self.apply_poses:
                        keypoint_poses_input = keypoint_poses[-n_obs_steps:].unsqueeze(0)
                        b, h, n = keypoint_poses_input.shape[:3]
                        keypoint_poses_input = F.pad(
                            keypoint_poses_input.reshape(b, h, n, -1),
                            (0, 0, 0, 0, npad, 0), mode='replicate'
                        ).view(b, -1, n, 4, 4)

                    obs_dict = dict()
                    if self.apply_rgb:
                        obs_dict["rgb"] = rgbs_input[:,-1]
                    if self.apply_pc:
                        obs_dict["pcd"] = pcds_input[:,-1]
                    if self.apply_low_dim_pcd:
                        obs_dict["low_dim_pcd"] = low_dim_pcds_input[:,-1]
                    if self.apply_poses:
                        obs_dict["keypoint_poses"] = keypoint_poses_input[:,-1]
                    obs_dict["agent_pose"] = gripper_input
                    obs_dict["low_dim_state"] = low_dim_state_input
                    
                    if self.adaptor is not None:
                        obs_dict = self.adaptor.adapt({'obs': obs_dict})['obs']
                    obs_dict = dict_apply(obs_dict, lambda x: x.type(dtype))
                    out = actioner.predict(obs_dict)
                    trajectory = out['rlbench_action']
                    
                    self.set_gripper_dummies(trajectory[:,:7])

                    if verbose:
                        print(f"Step {step_id}")

                    terminate = True
                    self.obs_history = []
                    # Update the observation based on the predicted action
                    try:
                        if verbose:
                            print("Plan with RRT")

                        for action, gripper_dummy in zip(trajectory[:self.n_action_steps], self._gripper_dummmies):
                            collision_checking = self._collision_checking(task_str, step_id)
                            observation, reward, terminate, _ = move(action, collision_checking=collision_checking)
                            self.set_gripper_renderable(gripper_dummy, False)

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

        log_data = {
            "success_rate": successful_demos / (len(demos) * demo_tries),
            "total_reward": total_reward,
            "rgbs_ls": rgbs_ls,
        }
        return log_data
    

    @torch.no_grad()
    def get_first_observation(self,       
                                task: TaskEnvironment,
                                task_str: str,
                                demos: List[Demo],  
                                max_steps: int,
                                actioner: Actioner,
                                max_rtt_tries: int = 1,
                                verbose: bool = False):
        self.launch()
        device = actioner.device
        dtype = actioner.dtype
        observations = []
        predictions = []
        n_obs_steps = self.n_obs_steps

        for demo_id, demo in enumerate(demos):
            if verbose:
                print()
                print(f"Starting demo {demo_id}")

        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        low_dim_pcds = torch.Tensor([]).to(device)
        keypoint_poses = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)
        low_dim_states = torch.Tensor([]).to(device)

        # descriptions, obs = task.reset()
        descriptions, obs = task.reset_to_demo(demo)
        self.obs_history.append(obs)

        move = Mover(task, max_tries=max_rtt_tries)
        reward = 0.0
        max_reward = 0.0

        for step_id in range(1):

            # Fetch the current observation, and predict one action
            if step_id > 0:
                self.obs_history = self.obs_history[::-1][::self.obs_history_augmentation_every_n][::-1][-n_obs_steps:]

            for observation in self.obs_history:
                obs_dict = self.get_obs_dict(observation)
                obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(device))
                
                if self.apply_rgb:
                    rgb = obs_dict['rgb'].to(device)
                    rgbs = torch.cat([rgbs, rgb], dim=0)
                if self.apply_pc:
                    pcd = obs_dict['pcd'].to(device)
                    pcds = torch.cat([pcds, pcd], dim=0)
                if self.apply_low_dim_pcd:
                    low_dim_pcd = obs_dict['low_dim_pcd'].to(device)
                    low_dim_pcds = torch.cat([low_dim_pcds, low_dim_pcd], dim=0)
                if self.apply_poses:
                    keypoint_pose = obs_dict['keypoint_poses'].to(device)
                    keypoint_poses = torch.cat([keypoint_poses, keypoint_pose], dim=0)

                gripper = obs_dict['gripper'].to(device)
                grippers = torch.cat([grippers, gripper], dim=0)
                low_dim_state = obs_dict['low_dim_state'].to(device)
                low_dim_states = torch.cat([low_dim_states, low_dim_state], dim=0)
            
            # Prepare proprioception history
            gripper_input = grippers[-n_obs_steps:].unsqueeze(0)

            npad = n_obs_steps - gripper_input.shape[1]
            gripper_input = F.pad(
                gripper_input, (0, 0, 0, 0, npad, 0), mode='replicate'
            )
            low_dim_state_input = low_dim_states[-n_obs_steps:].unsqueeze(0)
            low_dim_state_input = F.pad(
                low_dim_state_input, (0, 0, npad, 0), mode='replicate'
            )   
            if self.apply_rgb:
                rgbs_input = rgbs[-n_obs_steps:,:,:3].unsqueeze(0) # only rgb channels, remove attns
                b, _, n, c, h, w = rgbs_input.shape
                rgbs_input = F.pad(
                        rgbs_input.reshape(rgbs_input.shape[:2] + (-1,)),
                        (0, 0, npad, 0), mode='replicate'
                ).view(b, -1, n, c, h, w)

            if self.apply_pc:
                pcds_input = pcds[-n_obs_steps:].unsqueeze(0)
                b, _, n, c, h, w = pcds_input.shape
                pcds_input = F.pad(
                    pcds_input.reshape(pcds_input.shape[:2] + (-1,)),
                    (0, 0, npad, 0), mode='replicate'
                ).view(b, -1, n, c, h, w)

            if self.apply_low_dim_pcd:
                low_dim_pcds_input = low_dim_pcds[-n_obs_steps:].unsqueeze(0)
                low_dim_pcds_input = F.pad(
                    low_dim_pcds_input,
                    (0, 0, 0, 0, npad, 0), mode='replicate'
                )

            if self.apply_poses:
                keypoint_poses_input = keypoint_poses[-n_obs_steps:].unsqueeze(0)
                b, h, n = keypoint_poses_input.shape[:3]
                keypoint_poses_input = F.pad(
                    keypoint_poses_input.reshape(b, h, n, -1),
                    (0, 0, 0, 0, npad, 0), mode='replicate'
                ).view(b, -1, n, 4, 4)

            obs_dict = dict()
            if self.apply_rgb:
                obs_dict["rgb"] = rgbs_input[:,-1]
            if self.apply_pc:
                obs_dict["pcd"] = pcds_input[:,-1]
            if self.apply_low_dim_pcd:
                obs_dict["low_dim_pcd"] = low_dim_pcds_input[:,-1]
            if self.apply_poses:
                obs_dict["keypoint_poses"] = keypoint_poses_input[:,-1]
            obs_dict["agent_pose"] = gripper_input
            obs_dict["low_dim_state"] = low_dim_state_input

            if self.adaptor is not None:
                obs_dict = self.adaptor.adapt({'obs': obs_dict})['obs']
            obs_dict = dict_apply(obs_dict, lambda x: x.type(dtype))
            observations.append(obs_dict)
            action = actioner.predict(obs_dict)
            predictions.append(action)

        return observations, predictions
    
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
        task.reset_to_demo(demo)

        for i in range(demo_consistency_tries):
            gt_keyframe_actions = []
            gt_obs = []
            for f in keypoint_discovery(demo):
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
                    demo, task, max_rrt_tries, demo_consistency_tries
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

    @staticmethod
    def create_obs_config(
        image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
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
    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/keypoint/train"
    env = RLBenchEnv(
        data_path=data_path,
        headless=False,
        collision_checking=False,
        obs_history_augmentation_every_n=10,
        apply_rgb=False,
        apply_pc=False,
        apply_depth=False,
        apply_low_dim_pcd=True,
        render_image_size=[1280, 720]
    )

    task_str = "put_item_in_drawer"
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    variation = 0
    task.set_variation(variation)

    demos = env.get_demo(task_str, variation, episode_index=1)

    class ReplayPolicy:
        def __init__(self, demo):
            self._actions = get_actions_from_demo(demo)
            self.idx = 0
            self.n_obs_steps = 2

        def predict_action(self, obs):
            action = self._actions[self.idx % len(self._actions)]
            self.idx += 1
            return {
                'extra' :{
                "rlbench_action": action.unsqueeze(0),
                }
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


    logs = env._evaluate_task_on_demos(
        task_str=task_str,
        task=task,
        max_steps=100,
        demos=demos,
        actioner=replay_actioner,
        max_rtt_tries=10,
        demo_tries=1,
        n_visualize=1,
        verbose=True,
    )

    print('rgbs', logs['rgbs_ls'])

    def write_video(rgbs, path):
        import imageio.v2 as iio
        writer = iio.get_writer(path)
        for image in rgbs:
            writer.append_data(image.transpose(1, 2, 0))
        writer.close()

    write_video(logs["rgbs_ls"][0], "/home/felix/Workspace/diffusion_policy_felix/data/videos/rlbench_runner_test.mp4")

def load_config(config_name, overrides):
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg

def load_model(checkpoint_path, cfg):
    OmegaConf.resolve(cfg)
    policy = hydra.utils.instantiate(cfg.policy)
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["state_dicts"]['model'])
    return policy

def load_dataset(cfg):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset



def test():
    
    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/keypoint/train"
    from pytorch3d.transforms import matrix_to_quaternion, standardize_quaternion
    from diffusion_policy.policy.flow_matching_SE3_lowdim_policy import SE3FlowMatchingPolicy
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.common.common_utils import create_rlbench_action
    from diffusion_policy.common.adaptors import Peract2Robomimic
    TASK = "open_drawer"
    checkpoint_path = "/home/felix/Workspace/diffusion_policy_felix/data/outputs/2024.06.18/14.45.00_train_flow_matching_unet_lowdim_feat_pcd_open_drawer/checkpoints/latest.ckpt"

    cfg = load_config("train_flow_matching_SE3_lowdim_feat_pcd_workspace.yaml", [f"task={TASK}"])
    dataset = load_dataset(cfg)
    batch = dict_apply(dataset[0], lambda x: x.unsqueeze(0).float())

    policy : SE3FlowMatchingPolicy = load_model(checkpoint_path, cfg)
    policy.eval()


    action = policy.predict_action(batch['obs'])

    # visualize(batch['obs'], action["action"])

    adaptor = hydra.utils.instantiate(cfg.task.env_runner.adaptor)
    adaptor = Peract2Robomimic()
    env = RLBenchEnv(
        data_path=data_path,
        headless=True,
        collision_checking=False,
        n_action_steps=cfg.n_action_steps,
        obs_history_augmentation_every_n=cfg.task.env_runner.obs_history_augmentation_every_n,
        apply_rgb=False,
        apply_pc=False,
        apply_depth=False,
        apply_low_dim_pcd=True,
        render_image_size=[1280//2, 720//2],
        apply_pose=True,
        adaptor=adaptor,
    )

    task_str = "open_drawer_keypoint"
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    variation = 0
    task.set_variation(variation)

    actioner = Actioner(
        policy=policy,
        instructions={task_str: [[torch.rand(10)],[torch.rand(10)],[torch.rand(10)]]},
        action_dim=7,
    )

    sim_obs, sim_acts = env.get_first_observation(
        task = task,
        task_str = task_str,
        demos = dataset.demos[:1],
        max_steps=10,
        actioner=actioner,
    )

    compare_dicts(sim_obs[0], batch['obs'])
    print("Observations are consistent!")

    action_gt = create_rlbench_action(batch['action']['act_r'], batch['action']['act_p'], batch['action']['act_gr'], batch['action']['act_ic'])
    pred_dict = policy.predict_action(sim_obs[0])
    action_pred = create_rlbench_action(pred_dict['action']['act_r'], pred_dict['action']['act_p'], 
                                        pred_dict['action'].get('act_gr', None), pred_dict['action'].get('act_ic', None))

    print("GT action: ", action_pred)
    print("Pred action: ", action_pred)

    # print("GT pos:", batch['action']['act_p'])
    # print("Pred pos:", action_pred_dict['act_p'])

    # return

    eval_dict = policy.evaluate(batch)
    print_dict(eval_dict)




    logs = env._evaluate_task_on_demos(
        task_str=task_str,
        task=task,
        max_steps=4,
        demos=dataset.demos[:1],
        actioner=actioner,
        max_rtt_tries=10,
        demo_tries=1,
        n_visualize=1,
        verbose=True,
    )

    def write_video(rgbs, path):
        import imageio.v2 as iio
        writer = iio.get_writer(path, fps=30, format='FFMPEG', mode='I')
        for image in rgbs:
            writer.append_data(image.transpose(1, 2, 0))
        writer.close()

    write_video(logs["rgbs_ls"][0], "/home/felix/Workspace/diffusion_policy_felix/data/videos/rlbench_runner_test.mp4")


def test_dataset_simulation_consitency():
    from diffusion_policy.dataset.rlbench_dataset import RLBenchLowdimDataset
    from diffusion_policy.common.adaptors import Peract2Robomimic
    data_path = "/home/felix/Workspace/diffusion_policy_felix/data/keypoint/train"
    task_str = "stack_blocks"
    n_obs_steps = 2
    obs_history_augmentation_every_n=10
    dataset = RLBenchLowdimDataset(
        root=data_path,
        task_name=task_str,
        n_obs_steps=n_obs_steps,
        variation=0,
        num_episodes=1,
        obs_augmentation_every_n=obs_history_augmentation_every_n,
        use_low_dim_state=True,
        use_low_dim_pcd=False
    )

    sample0 = dict_apply(dataset[0], lambda x: x.unsqueeze(0).float())

    env = RLBenchEnv(
        data_path=data_path,
        headless=True,
        obs_history_augmentation_every_n=10,
        n_obs_steps=n_obs_steps,
        apply_rgb=False,
        apply_pc=False,
        apply_depth=False,
        apply_low_dim_pcd=False,
        render_image_size=[1280//2, 720//2],
        apply_pose=True,
        adaptor=Peract2Robomimic()
    )

    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)

    sim_obs, sim_acts = env.get_first_observation(
        task = task,
        task_str = task_str,
        demos = dataset.demos[:1],
        max_steps=10,
        actioner=sample0['obs']['robot0_eef_pos'],
    )

    print("Dataset obs")
    print_dict(sample0['obs'])
    print()
    print("Simulation obs")
    print_dict(sim_obs[0])
    print()
    print("Adapted simulation obs")
    print_dict(Peract2Robomimic().unadapt({'obs': sim_obs[0]})['obs'])

    compare_dicts(sim_obs[0], sample0['obs'])






if __name__ == "__main__":
    test()
    # test_dataset_simulation_consitency()
    print("Done!")