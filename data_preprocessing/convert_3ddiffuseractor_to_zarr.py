import os
import numpy as np
import torch
import zarr
from absl import app
from absl import flags
import pickle
import blosc
from pickle import UnpicklingError
from pathlib import Path
from collections import defaultdict

def loader(file):
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


FLAGS = flags.FLAGS
flags.DEFINE_string('save_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/diffuser_actor.zarr',
                    'Where to save the dataset.')
flags.DEFINE_string('data_path',
                    os.environ['DIFFUSION_POLICY_ROOT'] + '/data/Peract_packaged',
                    'Path to the data folder.')

def add_groups_to_demo(demo_group, cameras):
        camera_group = demo_group.create_group('cameras')
        for camera in cameras:
            camera_group.create_group(camera)
            camera_group[camera].create_dataset(f'rgb', shape=(0, 3, 256, 256), dtype=np.uint8, chunks=(1, 3, 256, 256))
            camera_group[camera].create_dataset(f'pcd', shape=(0, 3, 256, 256), dtype=np.float32, chunks=(1, 3, 256, 256))
            
        state_action_group = demo_group.create_group('state_action')
        state_action_group.create_dataset('proprioception', shape=(0, 7 + 1), dtype=np.float32, chunks=(1, 8))
        # describe the layout of the state_action_group with type and shape
        state_action_group.attrs['layout'] = 'gripper_pose : quat (7,), gripper_open : int (1,), ignore_collisions : int (1,)'

def convert_episode(episode):
    """
    the episode item: [
        [frame_ids],  # we use chunk and max_episode_length to index it
        [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
            obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
        [action_tensors],  # wrt frame_ids, (1, 8)
        [camera_dicts],
        [gripper_tensors],  # wrt frame_ids, (1, 8)
        [trajectories]  # wrt frame_ids, (N_i, 8)
    ]
    """
    frame_ids, obs_tensors, action_tensors, camera_dicts, gripper_tensors, trajectories = episode
    if isinstance(obs_tensors, torch.Tensor):
        obs_tensors = obs_tensors.numpy()

    rgb = obs_tensors[:, :, 0] * 255.0
    rgb = rgb.astype(np.uint8)
    pcd = obs_tensors[:, :, 1]
    cameras = camera_dicts[0].keys()
    rgb = {cam: rgb[:, i] for i, cam in enumerate(cameras)}
    pcd = {cam: pcd[:, i] for i, cam in enumerate(cameras)}

    # add one dummy observation for the last frame that is unused
    rgb = {cam: np.concatenate([rgb[cam], np.zeros_like(rgb[cam][:1])], axis=0) for cam in cameras}
    pcd = {cam: np.concatenate([pcd[cam], np.zeros_like(pcd[cam][:1])], axis=0) for cam in cameras}

    action_tensors = torch.cat(action_tensors)
    gripper_tensors = torch.cat(gripper_tensors)
    proprioception = torch.cat([gripper_tensors, action_tensors[-1:]]).numpy()

    return {
        'rgb': rgb,
        'pcd': pcd,
        'proprioception': proprioception,
    }


def main(argv):

    save_path = Path(FLAGS.save_path)
    data_path = Path(FLAGS.data_path)

    root = zarr.open(FLAGS.save_path, mode='w') 

    for split in ['train', 'val']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        split = root.create_group(split)

        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        taskvars = split_dir.glob('*')
        for taskvar_dir in taskvars:
            if not taskvar_dir.is_dir():
                continue
            task, var = taskvar_dir.name.split("+")
            if not taskvar_dir.is_dir():
                print(f"Can't find dataset folder {taskvar_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in taskvar_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in taskvar_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in taskvar_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {taskvar_dir}")
                continue
            episodes_by_task[task] += episodes
            print(f"Loaded {taskvar_dir.name} with {len(episodes)} episodes")
        
        for task, episodes in episodes_by_task.items():
            task_group = split.create_group(task)
            for i, (task, var, episode) in enumerate(episodes):
                # if var group does not exist, create it otherwise load it
                if var not in task_group:
                    var_group = task_group.create_group(var)
                else:
                    var_group = task_group[var]
                episode_data = loader(episode)
                if episode_data is None:
                    print(f"Can't load episode {episode}")
                    continue
                print(f"Loaded {episode}")
                episode_data = convert_episode(episode_data)
                demo_group = var_group.create_group(f'demo_{i}')
                add_groups_to_demo(demo_group, episode_data['rgb'].keys())
                for camera, rgb in episode_data['rgb'].items():
                    demo_group['cameras'][camera]['rgb'].append(rgb)
                for camera, pcd in episode_data['pcd'].items():
                    demo_group['cameras'][camera]['pcd'].append(pcd)
                demo_group['state_action']['proprioception'].append(episode_data['proprioception'])



if __name__ == '__main__':
  app.run(main)
