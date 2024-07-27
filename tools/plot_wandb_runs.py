import wandb
from typing import List
import tap
import matplotlib.pyplot as plt
import os

target_to_method_name_dict = {
    'diffusion_policy.policy.diffuser_actor.DiffuserActor': 'baseline',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_v2.DiffuserActor': 'ours',
    'diffusion_policy.policy.diffuser_actor_lowdim.DiffuserActor': 'baseline',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_lowdim.DiffuserActor': 'pose invariant',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_lowdim_v2.DiffuserActor': 'point invariant',
}

metric_to_title_dict = {
    'val_rotation_mse_error': 'Rotation MSE [val]',
    'val_position_mse_error': 'Position MSE [val]',
    'train_rotation_mse_error': 'Rotation MSE [train]',
    'train_position_mse_error': 'Position MSE [train]',
    'train_loss': 'Loss [train]',
    'val_loss': 'Loss [val]',
}

metric_to_ylabel_dict = {
    'val_rotation_mse_error': 'Rotation MSE',
    'val_position_mse_error': 'Position MSE',
    'train_rotation_mse_error': 'Rotation MSE',
    'train_position_mse_error': 'Position MSE',
    'train_loss': 'Loss',
    'val_loss': 'Loss',
}

def target_to_method_name(target, mask):
    method = target_to_method_name_dict[target]
    if mask:
        method += " [mask]"
    return method

class Arguments(tap.Tap):
    ids_path : str = "/home/felix/Workspace/diffusion_policy_felix/tools/highdim_wandb_ids.txt"
    username : str = "felix-herrmann"
    project_name : str = "diffusion_policy_debug"
    save_dir : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'wandb_plots')

def main(args):
    username = args.username
    project_name = args.project_name
    ids_path = args.ids_path
    save_dir = args.save_dir

    with open(ids_path, 'r') as file:
        content = file.read()
        ids = content.splitlines()

    os.makedirs(save_dir, exist_ok=True)

    wandb.login()
    api = wandb.Api()

    metrics = ['val_rotation_mse_error', 
               'val_position_mse_error',
                'train_rotation_mse_error',
                'train_position_mse_error',
                'train_loss',
                'val_loss']

    histories = {}
    cfgs = {}
    for id in ids:
        run = api.run(f"{username}/{project_name}/{id}")
        task_name = run.config['task']['task_name']
        lowdim = run.config['task']['type'] == 'lowdim'
        task_name += "_lowdim" if lowdim else ""
        if task_name not in histories:
            histories[task_name] = {}
            cfgs[task_name] = {}
        histories[task_name][id] = run.history(samples=10000, x_axis='epoch', keys=metrics)
        cfgs[task_name][id] = run.config
        use_mask = run.config['policy'].get('use_mask', False)
        method = target_to_method_name(run.config['policy']['_target_'], use_mask)
        print('Run:', id, task_name, method)

    for task in histories.keys():
        exp_path = os.path.join(save_dir, task)
        os.makedirs(exp_path, exist_ok=True)
        for metric in metrics:
            fig, ax = plt.subplots()
            for id, history in histories[task].items():
                cfg = cfgs[task][id]
                task_name = cfg['task']['name']
                use_mask = run.config['policy'].get('use_mask', False)
                method = target_to_method_name(cfg['policy']['_target_'], use_mask)
                data = history[[metric, 'epoch']]
                data = data.dropna()
                ax.plot(data['epoch'], data[metric], label=method)
            ax.set(xlabel='epoch', ylabel=metric_to_ylabel_dict[metric], title=metric_to_title_dict[metric] + " - " + task)
            ax.grid()
            ax.set_yscale('log')
            ax.legend()
            plt.savefig(os.path.join(exp_path, f'{metric}_{task}.png'))
            plt.close(fig)


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
