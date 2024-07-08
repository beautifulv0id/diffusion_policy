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

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/home/felix/Workspace/diffusion_policy_felix/data/image.zarr',
                    'Where to save the dataset.')
flags.DEFINE_string('data_path',
                    '/home/felix/Workspace/diffusion_policy_felix/data/image',
                    'Path to the data folder.')
flags.DEFINE_integer('n_demos', -1, 'Number of demos to use.')
flags.DEFINE_list('tasks', ['open_drawer', 'put_item_in_drawer', 'stack_blocks', 'sweep_to_dustpan_of_size', 'turn_tap'], 'Tasks to use.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')


def write_rlbench_dataset():

    num_demos = FLAGS.n_demos
    data_path = FLAGS.data_path
    dataset = zarr.open(FLAGS.save_path, mode='w')
    for task in FLAGS.tasks:
        task_path = os.path.join(data_path, FLAGS.tasks[0])
        episodes_path = os.path.join(task_path, 'variation0', 'episodes')

        if num_demos == -1:
            num_demos = len(os.listdir(episodes_path))
            
        # Create a new dataset

        task_group = dataset.create_group(task)

        # Create the data group
        data_group = task_group.create_group('data')

        # create arrays for the observations
        for camera in CAMERAS:
            data_group.create_dataset(f'{camera}_rgb', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))
            data_group.create_dataset(f'{camera}_mask', shape=(0, 1, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.uint8, chunks=(1, FLAGS.image_size[0], FLAGS.image_size[1]))
            data_group.create_dataset(f'{camera}_point_cloud', shape=(0, 3, FLAGS.image_size[0], FLAGS.image_size[1]), dtype=np.float32, chunks=(1, 3, FLAGS.image_size[0], FLAGS.image_size[1]))

        data_group.create_dataset('gripper_pose', shape=(0, 7), dtype=np.float32, chunks=(1, 7))
        data_group.create_dataset('ignore_collisions', shape=(0, 1), dtype=np.bool_, chunks=(1, 1))
        data_group.create_dataset('gripper_open', shape=(0, 1), dtype=np.bool_, chunks=(1, 1))
        data_group.create_dataset('gripper_joint_positions', shape=(0, 2), dtype=np.float32, chunks=(1, 2))

        # Create the meta group
        meta_group = task_group.create_group('meta')
        meta_group.create_dataset('episode_ends', shape=(0,), dtype=np.int32, chunks=(1,))    
        meta_group.create_dataset('keypoints', shape=(0,), dtype=np.int32, chunks=(1,))
        meta_group.create_dataset('keypoint_ends', shape=(0,), dtype=np.int32, chunks=(1,))
        meta_group.create_dataset('demos', shape=(num_demos,), dtype=object, object_codec=numcodecs.pickles.Pickle())
        # demos = zarr.empty(num_demos, dtype=object, object_codec=numcodecs.pickles.Pickle())
        
        obs_config = create_obs_config(image_size=FLAGS.image_size, apply_cameras=CAMERAS, apply_pc=True, apply_mask=True, apply_rgb=True, apply_depth=False)
        episode_end = 0
        keypoint_end = 0
        for demo_idx in range(num_demos):
            demo = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=False, dataset_root=FLAGS.data_path, random_selection=False, obs_config=obs_config)[0]
            keypoints = _keypoint_discovery(demo)
            keypoints = episode_end + np.array(keypoints, dtype=np.int32)
            meta_group['keypoints'].append(keypoints)
            episode_end = episode_end + len(demo)
            meta_group['episode_ends'].append(np.array([episode_end], dtype=np.int32))
            keypoint_end = keypoint_end + len(keypoints)
            meta_group['keypoint_ends'].append(np.array([keypoint_end], dtype=np.int32))
            for obs in demo:
                for camera in CAMERAS:
                    data_group[f'{camera}_rgb'].append(obs.__dict__[f"{camera}_rgb"].transpose(2, 0, 1)[None,...])
                    data_group[f'{camera}_point_cloud'].append(obs.__dict__[f"{camera}_point_cloud"].transpose(2, 0, 1)[None,...])
                    mask = (obs.__dict__[f"{camera}_mask"] > 97).astype(np.bool_)
                    data_group[f'{camera}_mask'].append(mask[None,None,...])

                data_group['gripper_pose'].append(obs.gripper_pose[None,...])
                data_group['gripper_open'].append(np.array([obs.gripper_open], dtype=np.bool_)[None,...])
                data_group['ignore_collisions'].append(np.array([obs.ignore_collisions], dtype=np.bool_)[None,...])
                data_group['gripper_joint_positions'].append(obs.gripper_joint_positions[None,...])
            demo._observations = []
            meta_group['demos'][demo_idx] = demo

def read_zarr_dataset():

    dataset = zarr.open(FLAGS.save_path, mode='r')

    print(dataset.tree())


def main(argv):
  write_rlbench_dataset()
  read_zarr_dataset()
  
   
if __name__ == '__main__':
  app.run(main)

