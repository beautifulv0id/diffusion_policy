import zarr
import numpy as np
from absl import app
from absl import flags
import os
from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
from diffusion_policy.common.rlbench_util import CAMERAS, create_obs_config, get_task_num_low_dim_pcd
from rlbench.utils import get_stored_demos
from diffusion_policy.common.rlbench_util import _keypoint_discovery
import numcodecs
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/lowdim_test_keypoints.zarr',
                    'Where to save the dataset.')
flags.DEFINE_string('data_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/image',
                    'Path to the data folder.')
flags.DEFINE_integer('n_demos', -1, 'Number of demos to use.')
flags.DEFINE_list('tasks', ['put_item_in_drawer', 'stack_blocks'], 'Tasks to use.')
flags.DEFINE_string('num_objects_path', 
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/diffusion_policy/tasks/peract_tasks_num_lowdim_pcd.json', 
                    'Path to the number of objects in each task.')

def write_rlbench_dataset():

    num_demos = FLAGS.n_demos
    data_path = FLAGS.data_path
    dataset = zarr.open(FLAGS.save_path, mode='w')
    with tqdm(FLAGS.tasks, desc="outer",
                    leave=False) as gtask:

        for task in gtask:
            task_path = os.path.join(data_path, FLAGS.tasks[0])
            episodes_path = os.path.join(task_path, 'variation0', 'episodes')

            if num_demos == -1:
                num_demos = len(os.listdir(episodes_path))

            print(FLAGS.num_objects_path, task)

            npcd = get_task_num_low_dim_pcd(FLAGS.num_objects_path, task)
                
            # Create a new dataset

            task_group = dataset.create_group(task)

            # Create the data group
            data_group = task_group.create_group('data')


            data_group.create_dataset('gripper_pose', shape=(0, 7), dtype=np.float32, chunks=(1, 7))
            data_group.create_dataset('ignore_collisions', shape=(0, 1), dtype=np.bool_, chunks=(1, 1))
            data_group.create_dataset('gripper_open', shape=(0, 1), dtype=np.bool_, chunks=(1, 1))
            data_group.create_dataset('gripper_joint_positions', shape=(0, 2), dtype=np.float32, chunks=(1, 2))
            data_group.create_dataset('low_dim_pcd', shape=(0, npcd, 3), dtype=np.float32, chunks=(1, npcd, 3))

            # Create the meta group
            meta_group = task_group.create_group('meta')
            meta_group.create_dataset('demos', shape=(num_demos,), dtype=object, object_codec=numcodecs.pickles.Pickle())
            meta_group.create_dataset('keypoint_ends', shape=(0,), dtype=np.int32, chunks=(1,))

            obs_config_low_dim = create_obs_config(image_size=[], apply_cameras=[], apply_pc=False, apply_mask=False, apply_rgb=False, apply_depth=False)
            keypoint_end = 0
            with tqdm(range(num_demos), desc=f"Task: {task}",
                        leave=False) as tepoch:
                for demo_idx in tepoch:
                    demo = get_stored_demos(amount = 1, variation_number=0, task_name=task, from_episode_number=demo_idx, image_paths=True, dataset_root=FLAGS.data_path, random_selection=False, obs_config=obs_config_low_dim)[0]
                    keypoints = _keypoint_discovery(demo)
                    if 0 not in keypoints:
                        keypoints = [0] + keypoints
                    keypoint_end = keypoint_end + len(keypoints)
                    meta_group['keypoint_ends'].append(np.array([keypoint_end], dtype=np.int32))
                    for kp in keypoints:
                        obs = demo[kp]
                        print(obs.misc['low_dim_pcd'].shape)
                        data_group['low_dim_pcd'].append(obs.misc['low_dim_pcd'][None,...])
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

