import zarr
import numpy as np
from absl import app
from absl import flags
import os
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config, get_workspace_bounds, get_task_num_low_dim_pcd
from rlbench.utils import get_stored_demos
from diffusion_policy.common.rlbench_util import _keypoint_discovery
from tqdm import tqdm
import pickle
from diffusion_policy.model.vision.clip_wrapper import load_clip
from diffusion_policy.model.vision.dino_wrapper import get_dino_features
import torch
from rlbench.backend.const import LOW_DIM_PICKLE
import torch
import torch.nn.functional as F
import json

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
flags.DEFINE_list('tasks', ['sweep_to_dustpan_of_size','open_drawer'], 'Tasks to use.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_string('workspace_bounds_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_workspace_bounds.json', 
                    'Path to the number of objects in each task.')
flags.DEFINE_string('num_objects_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_tasks_num_lowdim_pcd.json',
                    'Path to the number of objects in each task.')
flags.DEFINE_list('fuse_cameras', ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                  'Cameras to fuse.')
flags.DEFINE_list('feature_res', ['res2'], 'Feature resolution to use for the fused cameras.')
flags.DEFINE_bool('precompute_features', False, 'Whether to precompute features.')


FEATURE_RES_TO_DSF = {'res1': 2, 'res2': 4}


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
        if not os.path.exists(data_path):
            continue
        print(f'Processing folder {data_path}')
        split_group = dataset.create_group(split)
        with tqdm(FLAGS.tasks, desc="outer",
                        leave=False) as gtask:

            for task in gtask:
                task_path = os.path.join(data_path, task)
                task_group = split_group.create_group(task)
                variations = [var[9:] for var in os.listdir(task_path) if var.startswith('variation')]
                for variation in variations:
                    variation_group = task_group.create_group(variation)
                    episodes_path = os.path.join(task_path, 'variation'+variation, 'episodes')

                    if FLAGS.n_demos == -1:
                        num_demos = len(os.listdir(episodes_path))

                    print(f"Task: {task}, Variation: {variation}, Number of episodes: {num_demos}")

                    # Create a new dataset
                    # os.makedirs(os.path.join(save_path, task), exist_ok=True)

                    # create arrays for the observations
                    obs_config = create_obs_config(image_size=FLAGS.image_size, apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
                    obs_config_low_dim = create_obs_config(image_size=FLAGS.image_size, apply_cameras=[], apply_pc=False, apply_mask=False, apply_rgb=False, apply_depth=False)
                    with tqdm(range(num_demos), desc=f"Task: {task}",
                                leave=False) as tepoch:
                        for demo_idx in tepoch:
                            demo_lowdim = get_stored_demos(amount = 1, variation_number=int(variation), task_name=task, from_episode_number=demo_idx, image_paths=True, dataset_root=data_path, random_selection=False, obs_config=obs_config_low_dim)[0]
                            keypoints = _keypoint_discovery(demo_lowdim)
                            if 0 not in keypoints:
                                keypoints = [0] + keypoints
                            try:
                                demo = get_stored_demos(amount = 1, variation_number=int(variation), task_name=task, from_episode_number=demo_idx, image_paths=False, dataset_root=data_path, random_selection=False, obs_config=obs_config, obs_idxs=keypoints)[0]
                                demo_group = variation_group.create_group(f'demo_{demo_idx}')
                                add_groups_to_demo(demo_group, feature_map_pyramid)
                                if 'low_dim_pcd' in demo._observations[0].misc:
                                    npcd = get_task_num_low_dim_pcd(FLAGS.num_objects_path, task)
                                    demo_group.create_dataset('low_dim_pcd', shape=(0, npcd, 3), chunks=(1, npcd, 3))
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

                                        rgb = rgb / 255.0

                                        if FLAGS.precompute_features:
                                            # Extract DINO features
                                            # features = get_dino_features(rgb[0].transpose(1, 2, 0), scale=1)
                                            # features = features.cpu().numpy().transpose(2, 0, 1)[None,...]
                                            # camera_group[camera]['features']['dino_features'].append(features)

                                            # Extract CLIP features
                                            rgb = torch.tensor(rgb, device='cuda').float()
                                            with torch.no_grad():
                                                rgb_ = normalize(rgb)
                                                features = model(rgb_)
                                            for pyramid_lvl in feature_map_pyramid:
                                                camera_group[camera]['features']['clip_features'][pyramid_lvl['res']].append(features[pyramid_lvl['res']].cpu().numpy())
                                    
                                    proprioception = np.concatenate([obs.gripper_pose, [obs.gripper_open], [obs.ignore_collisions]])
                                    state_action_group['proprioception'].append(proprioception[None,...])
                                    if 'low_dim_pcd' in obs.misc:
                                        demo_group['low_dim_pcd'].append(obs.misc['low_dim_pcd'][None,...])
                                demo._observations = []
                                with open(os.path.join(save_root, demo_group.path, LOW_DIM_PICKLE), 'wb') as f:
                                    pickle.dump(demo, f)
                            except Exception as e:
                                print(f"Error processing split {split}, task {task}, demo {demo_idx}: {e}")
                                continue

def write_min_pcd_size(pcd_min):
    path = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], "diffusion_policy", "tasks", "peract_cropped_pcd_min_sizes.json")
    with open(path, 'w') as f:
        json.dump(pcd_min, f, indent=4)

def add_fused_camera_data():
    save_root = FLAGS.save_path
    cameras_to_fuse = FLAGS.fuse_cameras
    workspace_bounds = get_workspace_bounds(FLAGS.workspace_bounds_path)
    dataset = zarr.open(save_root, mode='a')

    # Define helper functions
    def interpolate_pcds(pcds, dsf):
        t, v, c, w, h = pcds.shape
        pcds_torch = torch.tensor(pcds).reshape(-1, c, w, h)
        pcds_torch = F.interpolate(pcds_torch, size=(w//dsf, h//dsf), mode='bilinear')
        pcds = pcds_torch.numpy()
        pcds = pcds.reshape(t, v, c, -1).transpose(0, 1, 3, 2).reshape(t, -1, c)
        return pcds
    
    def get_stacked_pcds(cameras_group, cameras):
        pcds = []
        for camera in cameras:
            camera_group = cameras_group[camera]
            pcds.append(camera_group['pcd'][:])
        pcds = np.stack(pcds, axis=1)
        return pcds
    
    def get_stacked_clip_features(cameras_group, cameras, res):
        clip_features = []
        for camera in cameras:
            camera_group = cameras_group[camera]
            clip_features.append(camera_group['features']['clip_features'][res][:])
        clip_features = np.stack(clip_features, axis=1)
        return clip_features
    
    def get_stacked_dino_features(cameras_group, cameras):
        dino_features = []
        for camera in cameras:
            camera_group = cameras_group[camera]
            dino_features.append(camera_group['features']['dino_features'][:])
        dino_features = np.stack(dino_features, axis=1)
        return dino_features


    def compute_totle_min_pcd_size(dataset, workspace_bounds):
        pcd_min_dict = {task: {res: np.inf for res in FLAGS.feature_res} for task in FLAGS.tasks}
        for res in FLAGS.feature_res:
            for split in FLAGS.splits:
                split_group = dataset[split]
                for task in FLAGS.tasks:
                    task_group = split_group[task]
                    for demo in task_group.keys():
                        demo_group = task_group[demo]
                        pcds = get_stacked_pcds(demo_group['cameras'], cameras_to_fuse)
                        pcds = interpolate_pcds(pcds, FEATURE_RES_TO_DSF[res])
                        min_ = np.min([crop_workspace(pcd, workspace_bounds=workspace_bounds).shape[0] for pcd in pcds])
                        pcd_min_dict[task][res] = int(np.min([pcd_min_dict[task][res], min_]))
        return pcd_min_dict
    
    pcd_min_dict = compute_totle_min_pcd_size(dataset, workspace_bounds)
    write_min_pcd_size(pcd_min_dict)

                                                   
    for split in FLAGS.splits:
        split_group = dataset[split]
        for task in FLAGS.tasks:
            task_group = split_group[task]
            for demo in task_group.keys():
                demo_group = task_group[demo]
                if 'fused_cameras' in demo_group:
                    del demo_group['fused_cameras']
                fused_cameras = demo_group.create_group('fused_cameras')
                for res in FLAGS.feature_res:
                    pcd_min = pcd_min_dict[task][res]
                    res_group = fused_cameras.create_group(res)
                    pcds = get_stacked_pcds(demo_group['cameras'], cameras_to_fuse)
                    clip_features = get_stacked_clip_features(demo_group['cameras'], cameras_to_fuse, res)
                    pcds = interpolate_pcds(pcds, FEATURE_RES_TO_DSF[res])

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

                    res_group.create_dataset('pcd', data=pcds, chunks=(1, pcd_min, 3))
                    res_group.create_dataset('clip_features', data=clip_features, chunks=(1, pcd_min, clip_features.shape[-1]))


def read_zarr_dataset():

    dataset = zarr.open(FLAGS.save_path, mode='r')

    print(dataset.tree())

def main(argv):
    write_rlbench_dataset()
    if FLAGS.precompute_features:
        add_fused_camera_data()
    read_zarr_dataset()
  
   
if __name__ == '__main__':
  app.run(main)

