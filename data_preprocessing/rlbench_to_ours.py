import zarr
import numpy as np
from absl import app
from absl import flags
import os
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config
from rlbench.utils import get_stored_demos
from diffusion_policy.common.rlbench_util import _keypoint_discovery
import numcodecs
from tqdm import tqdm
from PIL import Image
import json
from diffusion_policy.model.vision.clip_wrapper import load_clip
import torch

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
flags.DEFINE_list('tasks', ['open_drawer'], 'Tasks to use.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_string('num_objects_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_tasks_num_lowdim_pcd.json', 
                    'Path to the number of objects in each task.')

def save_extrinsics_intrinsics_to_json(path, extrinsics, intrinsics):
    data = { 
        "extrinsics": extrinsics.tolist(),
        "intrinsics": intrinsics.tolist()
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

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

                # create arrays for the observations
                camera_group = task_group.create_group('cameras')
                for camera in CAMERAS:
                    camera_group.create_group(camera)
                    camera_group[camera].create_dataset(f'rgb', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))
                    feature_group = camera_group[camera].create_group('features')
                    for pyramid_lvl in feature_map_pyramid:
                        feature_group.create_dataset(pyramid_lvl['res'], shape=(0, pyramid_lvl['size'], FLAGS.image_size[0]//pyramid_lvl['dsf'], FLAGS.image_size[1]//pyramid_lvl['dsf']), dtype=np.uint8, chunks=(1, 256, FLAGS.image_size[0]//pyramid_lvl['dsf'], FLAGS.image_size[1]//pyramid_lvl['dsf']))
                    camera_group[camera].create_dataset(f'mask', shape=(0, 1, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, FLAGS.image_size[0], FLAGS.image_size[1]))
                    camera_group[camera].create_dataset(f'pcd', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.float32, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))

                state_action_group = task_group.create_group('state_action')
                state_action_group.create_dataset('proprioception', shape=(0, 7 + 1 + 1), dtype=np.float32, chunks=(1, 9))
                # describe the layout of the state_action_group with type and shape
                state_action_group.attrs['layout'] = 'gripper_pose : quat (7,), gripper_open : int (1,), ignore_collisions : int (1,)'

                # Create the meta group
                meta_group = task_group.create_group('meta')
                meta_group.create_dataset('demos', shape=(num_demos,), dtype=object, object_codec=numcodecs.pickles.Pickle())
                meta_group.create_dataset('keypoint_ends', shape=(0,), dtype=np.int32, chunks=(1,))

                os.makedirs(os.path.join(save_path, task), exist_ok=True)

                # create arrays for the observations
                obs_config = create_obs_config(image_size=FLAGS.image_size, apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
                obs_config_low_dim = create_obs_config(image_size=FLAGS.image_size, apply_cameras=[], apply_pc=False, apply_mask=False, apply_rgb=False, apply_depth=False)
                keypoint_end = 0
                with tqdm(range(num_demos), desc=f"Task: {task}",
                            leave=False) as tepoch:
                    for demo_idx in tepoch:
                        demo_lowdim = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=True, dataset_root=data_path, random_selection=False, obs_config=obs_config_low_dim)[0]
                        keypoints = _keypoint_discovery(demo_lowdim)
                        if 0 not in keypoints:
                            keypoints = [0] + keypoints
                        demo = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=False, dataset_root=data_path, random_selection=False, obs_config=obs_config, obs_idxs=keypoints)[0]
                        keypoint_end = keypoint_end + len(keypoints)
                        meta_group['keypoint_ends'].append(np.array([keypoint_end], dtype=np.int32))
                        
                        for kp in keypoints:
                            obs = demo[kp]
                            for camera in CAMERAS:
                                rgb = obs.__dict__[f"{camera}_rgb"].transpose(2, 0, 1)[None,...]
                                camera_group[camera]['rgb'].append(rgb)
                                camera_group[camera]['pcd'].append(obs.__dict__[f"{camera}_point_cloud"].transpose(2, 0, 1)[None,...])
                                mask = (obs.__dict__[f"{camera}_mask"] > 97).astype(np.bool_)
                                camera_group[camera][f'mask'].append(mask[None,None,...])

                                # Extract features
                                rgb = torch.tensor(rgb, device='cuda').float() / 255.0
                                with torch.no_grad():
                                    rgb = normalize(rgb)
                                    features = model(rgb)
                                for pyramid_lvl in feature_map_pyramid:
                                    camera_group[camera]['features'][pyramid_lvl['res']].append(features[pyramid_lvl['res']].cpu().numpy())
                            
                            proprioception = np.concatenate([obs.gripper_pose, [obs.gripper_open], [obs.ignore_collisions]])
                            state_action_group['proprioception'].append(proprioception[None,...])
                        demo._observations = []
                        meta_group['demos'][demo_idx] = demo

def main(argv):
  write_rlbench_dataset()
  
   
if __name__ == '__main__':
  app.run(main)

