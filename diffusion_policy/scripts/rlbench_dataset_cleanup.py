from diffusion_policy.env.rlbench.rlbench_lowdim_env import RLBenchLowDimEnv
import os
import click

def remove_unvalid_demos(data_path, task_str, variation):
    env = RLBenchLowDimEnv(
        data_path=data_path,
        headless=True,
        collision_checking=False,
    )

    variation_str = f"variation{variation}" 
    if variation == -1:
        variation_str = "all_variations"

    episodes_path = os.path.join(data_path, task_str, variation_str,"episodes")
    episode_path = os.path.join(episodes_path, "episode%d")
    unvalid_episode_path = os.path.join(episodes_path, "unvalid_episode%d")
    n_demos = len([path for path in os.listdir(episodes_path) if path.startswith("episode")])
    sr, demo_valid, success_rates = env.verify_demos(task_str, variation, n_demos, max_tries=10, demo_consistency_tries=10, verbose=False)
    valid_episode_idx = 0
    for i, val in enumerate(demo_valid):
        if not val:
            os.rmdir(episode_path % i)
        else:
            os.rename(episode_path % i, episode_path % valid_episode_idx)
            valid_episode_idx += 1       

    print(f"Success rates: {success_rates}")

@click.command()
@click.option(
    '--data_path', '-d', required=True,
    help='Data search path'
)
@click.option(
    '--task_str', '-t', required=True,
    help='Task string'
)
@click.option(
    '--variation', '-v', 
    type=int,
    required=True,
    help='Variation number'
)
def main(data_path, task_str, variation):
    remove_unvalid_demos(data_path, task_str, variation)

if __name__ == '__main__':
    main()
    print("Done")