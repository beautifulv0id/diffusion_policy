import zarr
import numpy as np
from absl import app
from absl import flags
import os
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config, get_workspace_bounds
from rlbench.utils import get_stored_demos
from diffusion_policy.common.rlbench_util import _keypoint_discovery
from tqdm import tqdm
import pickle
from diffusion_policy.model.vision.clip_wrapper import load_clip
from diffusion_policy.model.vision.dino_wrapper import get_dino_features
import torch
from rlbench.backend.const import LOW_DIM_PICKLE
import skimage.transform

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/peract.zarr',
                    'Where to save the dataset.')
flags.DEFINE_string('data_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/peract',
                    'Path to the data folder.')
flags.DEFINE_list('splits', ['train', 'val', 'test'],
                    'Splits to use.')
flags.DEFINE_integer('n_demos', -1, 'Number of demos to use.')
flags.DEFINE_list('tasks', ['open_drawer', 'sweep_to_dustpan_of_size'], 'Tasks to use.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_string('workspace_bounds_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_workspace_bounds.json', 
                    'Path to the number of objects in each task.')
flags.DEFINE_list('fuse_cameras', ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                  'Cameras to fuse.')


def crop_workspace(pcd, workspace_bounds=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]):
    """
    Filters points and RGB values within the specified workspace bounds (no batch dimension).

    Parameters:
    - pcd: A numpy array of shape (N, 3) representing the point cloud.
    - workspace_bounds: A numpy array of shape (2, 3) specifying the min and max bounds for x, y, z.

    Returns:
    - indices: A numpy array of shape (N,) containing the indices of the points within the workspace.
    """
    mask = np.ones(pcd.shape[0], dtype=bool)
    for i in range(3):
        mask = np.logical_and(mask, pcd[:, i] > workspace_bounds[0, i])
        mask = np.logical_and(mask, pcd[:, i] < workspace_bounds[1, i])

    indices = np.where(mask)[0]
    return indices

def add_groups_to_demo(demo_group, feature_map_pyramid):
        camera_group = demo_group.create_group('cameras')
        for camera in CAMERAS:
            camera_group.create_group(camera)
            camera_group[camera].create_dataset(f'rgb', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))
            feature_group = camera_group[camera].create_group('features')
            clip_features = feature_group.create_group('clip_features')
            for pyramid_lvl in feature_map_pyramid:
                clip_features.create_dataset(pyramid_lvl['res'], shape=(0, pyramid_lvl['size'], FLAGS.image_size[0]//pyramid_lvl['dsf'], FLAGS.image_size[1]//pyramid_lvl['dsf']), dtype=np.uint8, chunks=(1, 256, FLAGS.image_size[0]//pyramid_lvl['dsf'], FLAGS.image_size[1]//pyramid_lvl['dsf']))
            feature_group.create_dataset('dino_features', shape=(0, 384, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.float32, chunks=(1, 384, FLAGS.image_size[0], FLAGS.image_size[1]))
            camera_group[camera].create_dataset(f'mask', shape=(0, 1, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, FLAGS.image_size[0], FLAGS.image_size[1]))
            camera_group[camera].create_dataset(f'pcd', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.float32, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))
            camera_group[camera].create_dataset(f'intrinsics', shape=(0, 3, 3), dtype=np.float32, chunks=(1, 3, 3))
            camera_group[camera].create_dataset(f'extrinsics', shape=(0, 4, 4), dtype=np.float32, chunks=(1, 4, 4))
            
        state_action_group = demo_group.create_group('state_action')
        state_action_group.create_dataset('proprioception', shape=(0, 7 + 1 + 1), dtype=np.float32, chunks=(1, 9))
        # describe the layout of the state_action_group with type and shape
        state_action_group.attrs['layout'] = 'gripper_pose : quat (7,), gripper_open : int (1,), ignore_collisions : int (1,)'

def write_rlbench_dataset():

    data_root = FLAGS.data_path
    save_root = FLAGS.save_path
    splits = FLAGS.splits
    feature_map_pyramid = [{'dsf': 2, 'res': 'res1', 'size': 64}, 
                            {'dsf': 4, 'res': 'res2', 'size': 256},]
    dataset = zarr.open(FLAGS.save_path, mode='w')
    model, normalize = load_clip()
    model.eval()
    model = model.to('cuda')

    for split in splits:
        data_path = os.path.join(data_root, split)
        save_path = os.path.join(save_root, split)
        print(f'Processing folder {data_path}')
        split_group = dataset.create_group(split)
        with tqdm(FLAGS.tasks, desc="outer",
                        leave=False) as gtask:

            for task in gtask:
                task_path = os.path.join(data_path, task)
                episodes_path = os.path.join(task_path, 'variation0', 'episodes')

                if FLAGS.n_demos == -1:
                    num_demos = len(os.listdir(episodes_path))

                print(f"Task: {task}, Number of episodes: {num_demos}")

                # Create a new dataset
                task_group = split_group.create_group(task)
                os.makedirs(os.path.join(save_path, task), exist_ok=True)

                # create arrays for the observations
                obs_config = create_obs_config(image_size=FLAGS.image_size, apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
                obs_config_low_dim = create_obs_config(image_size=FLAGS.image_size, apply_cameras=[], apply_pc=False, apply_mask=False, apply_rgb=False, apply_depth=False)
                with tqdm(range(num_demos), desc=f"Task: {task}",
                            leave=False) as tepoch:
                    for demo_idx in tepoch:
                        demo_lowdim = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=True, dataset_root=data_path, random_selection=False, obs_config=obs_config_low_dim)[0]
                        keypoints = _keypoint_discovery(demo_lowdim)
                        if 0 not in keypoints:
                            keypoints = [0] + keypoints
                        demo = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=False, dataset_root=data_path, random_selection=False, obs_config=obs_config, obs_idxs=keypoints)[0]
                        
                        demo_group = task_group.create_group(f'demo_{demo_idx}')
                        add_groups_to_demo(demo_group, feature_map_pyramid)
                        camera_group = demo_group['cameras']
                        state_action_group = demo_group['state_action']
                        for kp in keypoints:
                            obs = demo[kp]
                            for camera in CAMERAS:
                                rgb = obs.__dict__[f"{camera}_rgb"].transpose(2, 0, 1)[None,...]
                                camera_group[camera]['rgb'].append(rgb)
                                camera_group[camera]['pcd'].append(obs.__dict__[f"{camera}_point_cloud"].transpose(2, 0, 1)[None,...])
                                mask = (obs.__dict__[f"{camera}_mask"] > 97).astype(np.bool_)
                                camera_group[camera][f'mask'].append(mask[None,None,...])
                                camera_group[camera][f'intrinsics'].append(obs.misc[f"{camera}_camera_intrinsics"][None,...])
                                camera_group[camera][f'extrinsics'].append(obs.misc[f"{camera}_camera_extrinsics"][None,...])

                                # Extract DINO features
                                rgb = rgb / 255.0
                                features = get_dino_features(rgb[0].transpose(1, 2, 0), scale=1)
                                features = features.cpu().numpy().transpose(2, 0, 1)[None,...]
                                camera_group[camera]['features']['dino_features'].append(features)

                                # Extract CLIP features
                                rgb = torch.tensor(rgb, device='cuda').float()
                                with torch.no_grad():
                                    rgb_ = normalize(rgb)
                                    features = model(rgb_)
                                for pyramid_lvl in feature_map_pyramid:
                                    camera_group[camera]['features']['clip_features'][pyramid_lvl['res']].append(features[pyramid_lvl['res']].cpu().numpy())
                            
                            proprioception = np.concatenate([obs.gripper_pose, [obs.gripper_open], [obs.ignore_collisions]])
                            state_action_group['proprioception'].append(proprioception[None,...])
                        demo._observations = []
                        with open(os.path.join(save_root, demo_group.path, LOW_DIM_PICKLE), 'wb') as f:
                            pickle.dump(demo, f)

def add_fused_camera_data():
    downsampling_factor = 2 # use 2 for CLIP with res1 and 4 for CLIP with res2
    save_root = FLAGS.save_path
    cameras_to_fuse = FLAGS.fuse_cameras
    workspace_bounds = get_workspace_bounds(FLAGS.workspace_bounds_path)
    dataset = zarr.open(save_root, mode='a')

    # Define helper functions
    def interpolate_pcds(pcds, downsampling_factor):
        t, v, c, w, h = pcds.shape
        pcds = pcds.transpose(0, 1, 3, 4, 2) 
        pcds = pcds.reshape(t*v, w, h, c)
        pcds = np.stack([skimage.transform.resize(pcd, (h//downsampling_factor, w//downsampling_factor)).astype('float32') for pcd in pcds])
        pcds = pcds.reshape(t, -1, c)
        return pcds
    
    def get_stacked_pcd_and_features(cameras_group, cameras):
        pcds = []
        clip_features = []
        for camera in cameras:
            camera_group = cameras_group[camera]
            pcds.append(camera_group['pcd'][:])
            clip_features.append(camera_group['features']['clip_features']['res1'][:])
        pcds = np.stack(pcds, axis=1)
        clip_features = np.stack(clip_features, axis=1)
        return pcds, clip_features

    def compute_totle_min_pcd_size(dataset, workspace_bounds):
        min_pcd_len = np.inf
        for split in dataset.keys():
            split_group = dataset[split]
            for task in split_group.keys():
                task_group = split_group[task]
                for demo in task_group.keys():
                    demo_group = task_group[demo]
                    pcds, _ = get_stacked_pcd_and_features(demo_group['cameras'], cameras_to_fuse)
                    pcds = interpolate_pcds(pcds, downsampling_factor)
                    min_ = np.min([crop_workspace(pcd, workspace_bounds=workspace_bounds).shape[0] for pcd in pcds])
                    min_pcd_len = np.min([min_pcd_len, min_])
        return int(min_pcd_len)
    
    pcd_min = compute_totle_min_pcd_size(dataset, workspace_bounds)
                                                         
    for split in dataset.keys():
        split_group = dataset[split]
        for task in split_group.keys():
            task_group = split_group[task]
            for demo in task_group.keys():
                demo_group = task_group[demo]
                if 'fused_cameras' in demo_group:
                    del demo_group['fused_cameras']
                fused_cameras = demo_group.create_group('fused_cameras')
                pcds, clip_features = get_stacked_pcd_and_features(demo_group['cameras'], cameras_to_fuse)
                pcds = interpolate_pcds(pcds, downsampling_factor)

                # reshape clip features
                t, v, c, h, w = clip_features.shape
                clip_features = clip_features.transpose(0, 1, 3, 4, 2)
                clip_features = clip_features.reshape(t, -1, c)


                pcd_list = []
                clip_feature_list = []
                for i in range(len(pcds)):
                    indices = crop_workspace(pcds[i], workspace_bounds=workspace_bounds)
                    indices = np.random.choice(indices, pcd_min, replace=False)

                    pcd_list.append(pcds[i][indices])
                    clip_feature_list.append(clip_features[i][indices])

                pcds = np.stack(pcd_list)
                clip_features = np.stack(clip_feature_list)

                fused_cameras.create_dataset('pcd', data=pcds, chunks=(1, pcd_min, 3))
                fused_cameras.create_dataset('clip_features', data=clip_features, chunks=(1, pcd_min, clip_features.shape[-1]))


def read_zarr_dataset():

    dataset = zarr.open(FLAGS.save_path, mode='r')

    print(dataset.tree())

def main(argv):
    write_rlbench_dataset()
    add_fused_camera_data()
    read_zarr_dataset()
  
   
if __name__ == '__main__':
  app.run(main)

