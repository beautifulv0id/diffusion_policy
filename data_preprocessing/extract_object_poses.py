from diffusion_policy.env.rlbench.rlbench_lowdim_env import RLBenchLowDimEnv
from diffusion_policy.env.rlbench.rlbench_utils import task_file_to_task_class, get_object_pose_indices_from_task
import os
from subprocess import call
import pickle
from pathlib import Path
from pyrep.const import ObjectType
from os.path import join, exists
from os import listdir
import tap
from rlbench.backend.const import *
import numpy as np
import tqdm
import json

class Arguments(tap.Tap):
    root_dir: Path
    meta_out_dir: Path = Path("../diffusion_policy/tasks")

def main(root_dir, task_name): 
    env = RLBenchLowDimEnv(data_path=root_dir, 
                            headless=True, 
                            collision_checking=True)
    task_type = task_file_to_task_class(task_name)
    task_env = env.env.get_task(task_type)
    task = task_env._task

    object_idxs = get_object_pose_indices_from_task(task)

    task_root = join(root_dir, task_name)
    num_objects = object_idxs.shape[0]
    
    with open(join(task_root, 'num_objects.txt'), 'w') as f:
        f.write(str(num_objects))

    variations = listdir(task_root)
    variations = [v for v in variations if '.DS_Store' not in v]

    for variation in tqdm.tqdm(variations):
        examples_path = join(
            task_root, variation,
            EPISODES_FOLDER)
        if not os.path.isdir(examples_path):
            continue
        examples = listdir(examples_path)
        examples = [e for e in examples if '.DS_Store' not in e]
        # load demo pickle
        for example in examples:
            example_path = join(examples_path, example)
            if not exists(example_path):
                continue
            if os.path.islink(example_path):
                continue

            with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
                demo = pickle.load(f)

            for obs in demo:
                object_poses = []
                task_low_dim_state = obs.task_low_dim_state
                for idx in object_idxs:
                    object_poses.append(task_low_dim_state[idx:idx+7])
                object_poses = np.array(object_poses)
                obs.misc['object_poses'] = object_poses

            with open(join(example_path, LOW_DIM_PICKLE), 'wb') as f:
                pickle.dump(demo, f)

    env.env.shutdown()

    return num_objects

if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    out_dir = str(args.meta_out_dir.absolute())
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    task_num_objects_map = {}   
    for task in tasks:
        print(f'Processing {task}')
        n = main(root_dir, task)
        task_num_objects_map[task] = n

    with open(join(out_dir, 'peract_tasks_num_objects.json'), 'w') as f:
        json.dump(task_num_objects_map, f, indent=4)
