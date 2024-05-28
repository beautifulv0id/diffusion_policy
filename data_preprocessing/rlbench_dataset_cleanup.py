from diffusion_policy.env.rlbench.rlbench_env import RLBenchEnv
import os
from pathlib import Path
import tap
import shutil

class Arguments(tap.Tap):
    root_dir: Path
    task_str: str = None
    variation: int = -1

def remove_unvalid_demos(root_dir, task_str, variation):
    env = RLBenchEnv(
        data_path=root_dir,
        headless=True,
        collision_checking=False,
    )

    variation_str = f"variation{variation}" 
    if variation == -1:
        variation_str = "all_variations"

    episodes_path = os.path.join(root_dir, task_str, variation_str,"episodes")
    episode_path = os.path.join(episodes_path, "episode%d")
    unvalid_episode_path = os.path.join(episodes_path, "unvalid_episode%d")
    n_demos = len([path for path in os.listdir(episodes_path) if path.startswith("episode")])
    sr, demo_valid, success_rates = env.verify_demos(task_str, variation, n_demos, max_rrt_tries=10, demo_consistency_tries=10, verbose=False)
    valid_episode_idx = 0
    for i, val in enumerate(demo_valid):
        if not val:
            shutil.rmtree(episode_path % i)
        else:
            os.rename(episode_path % i, episode_path % valid_episode_idx)
            valid_episode_idx += 1       

    print(f"Success rates: {success_rates}")

def main(root_dir, task_str, variation):
    if task_str is None:
        root_dir = str(Path(root_dir).absolute())
        tasks = [f for f in os.listdir(root_dir) if ('.zip' not in f) and ('.DS_Store' not in f)]
        for task in tasks:
            print(f'Processing {task}')
            remove_unvalid_demos(root_dir, task, variation)
    else:
        remove_unvalid_demos(root_dir, task_str, variation)

if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args.root_dir, args.task_str, args.variation)
    print("Done")