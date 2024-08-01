import wandb
from typing import List
import tap
import matplotlib.pyplot as plt
import os
import numpy as np

target_to_method_name_dict = {
    'diffusion_policy.policy.diffuser_actor.DiffuserActor': 'baseline',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_v2.DiffuserActor': 'ours',
    'diffusion_policy.policy.diffuser_actor_lowdim.DiffuserActor': 'baseline',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_lowdim.DiffuserActor': 'PIA',
    'diffusion_policy.policy.diffuser_actor_pose_invariant_lowdim_v2.DiffuserActor': 'IPA',
}

metric_to_title_dict = {
    'val_rotation_mse_error': 'Rotation MSE [val]',
    'val_position_mse_error': 'Position MSE [val]',
    'train_rotation_mse_error': 'Rotation MSE [train]',
    'train_position_mse_error': 'Position MSE [train]',
    'train_loss': 'Loss [train]',
    'val_loss': 'Loss [val]',
    'train_success_rate': 'Success Rate [train]',
    'eval_success_rate': 'Success Rate [val]',
}

metric_to_ylabel_dict = {
    'val_rotation_mse_error': 'Rotation MSE',
    'val_position_mse_error': 'Position MSE',
    'train_rotation_mse_error': 'Rotation MSE',
    'train_position_mse_error': 'Position MSE',
    'train_loss': 'Loss',
    'val_loss': 'Loss',
    'train_success_rate': 'Success Rate',
    'eval_success_rate': 'Success Rate',
}

task_to_task_name_dict = {
    'open_drawer': 'Open Drawer',
    'put_item_in_drawer': 'Put Item in Drawer',
    'stack_blocks': 'Stack Blocks',
    'sweep_to_dustpan_of_size': 'Sweep to Dustpan of Size',
    'turn_tap': 'Turn Tap',
}

def task_name_to_paper_name(task_name):
    task = task_name.replace('_lowdim', '')
    return task_to_task_name_dict[task]

def target_to_method_name(target, mask):
    method = target_to_method_name_dict[target]
    if mask:
        method += " [mask]"
    return method

class Arguments(tap.Tap):
    ids_path : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'tools/highdim_wandb_ids.txt')
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
        lines = content.splitlines()
        ids = [line.split(' ')[0] for line in lines]

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
    summaries = {}
    cfgs = {}
    for id in ids:
        run = api.run(f"{username}/{project_name}/{id}")
        task_name = run.config['task']['task_name']
        type = run.config['task']['type']
        if type not in histories.keys():
            histories[type] = {}
            summaries[type] = {}
            cfgs[type] = {}

        if task_name not in histories[type]:
            histories[type][task_name] = {}
            cfgs[type][task_name] = {}
            summaries[type][task_name] = {}
        histories[type][task_name][id] = run.history(samples=10000, x_axis='epoch', keys=metrics)
        summaries[type][task_name][id] = run.summary
        cfgs[type][task_name][id] = run.config
        use_mask = run.config['policy'].get('use_mask', False)
        method = target_to_method_name(run.config['policy']['_target_'], use_mask)
        print('Run:', id, task_name, method)

    # for type in histories.keys():
    #     os.makedirs(os.path.join(save_dir, type), exist_ok=True)
    #     for task in histories[type].keys():
    #         exp_path = os.path.join(save_dir, type, task)
    #         os.makedirs(exp_path, exist_ok=True)
    #         for metric in metrics:
    #             fig, ax = plt.subplots()
    #             for id, history in histories[type][task].items():
    #                 cfg = cfgs[type][task][id]
    #                 task_name = cfg['task']['name']
    #                 use_mask = cfg['policy'].get('use_mask', False)
    #                 method = target_to_method_name(cfg['policy']['_target_'], use_mask)
    #                 data = history[[metric, 'epoch']]
    #                 data = data.dropna()
    #                 ax.plot(data['epoch'], data[metric], label=method)
    #                 ax.set(xlabel='epoch', ylabel=metric_to_ylabel_dict[metric], title=metric_to_title_dict[metric] + " - " + task)
    #                 ax.set_yscale('log')
    #             ax.grid()
    #             ax.legend()
    #             plt.savefig(os.path.join(exp_path, f'{metric}_{task_name_to_paper_name(task)}_{type}.png'))
    #             plt.close(fig)

    # barplots for success rates
    for metric in ['train_success_rate', 'eval_success_rate']:
        for type in histories.keys():
            tasks = histories[type].keys()
            data = {}
            exp_path = os.path.join(save_dir, type, 'barplots')
            os.makedirs(exp_path, exist_ok=True)
            multiplier = 0

            for task in histories[type].keys():
                for id, summary in summaries[type][task].items():
                    cfg = cfgs[type][task][id]
                    task_name = cfg['task']['name']
                    use_mask = cfg['policy'].get('use_mask', False)
                    method = target_to_method_name(cfg['policy']['_target_'], use_mask)
                    if method not in data:
                        data[method] = []
                    data[method].append(summary.get(metric, 0))
            fig, ax = plt.subplots(layout='constrained', figsize=(12, 5))
            x = np.arange(len(tasks))
            for method, values in data.items():
                width = 1 / len(values)
                offset = width * multiplier
                rects = ax.bar(x + offset, values, width=width, label=method)
                ax.bar_label(rects, padding=3)
                multiplier += 1

            ax.set(xlabel='Task', ylabel=metric_to_ylabel_dict[metric], title=metric_to_title_dict[metric])
            ax.grid(False)
            ax.legend()
            ax.set_xticks(x + width, [task_name_to_paper_name(task) for task in tasks])
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}.png'))
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}.svg'))
            plt.close(fig)

        
    # barplots for success rates
    for metric in ['val_rotation_mse_error', 
               'val_position_mse_error',
                'train_rotation_mse_error',
                'train_position_mse_error']:
        for type in histories.keys():
            tasks = histories[type].keys()
            data = {}
            exp_path = os.path.join(save_dir, type, 'barplots')
            os.makedirs(exp_path, exist_ok=True)
            multiplier = 0

            for task in histories[type].keys():
                for id, history in histories[type][task].items():
                    cfg = cfgs[type][task][id]
                    task_name = cfg['task']['name']
                    use_mask = cfg['policy'].get('use_mask', False)
                    method = target_to_method_name(cfg['policy']['_target_'], use_mask)
                    if method not in data:
                        data[method] = []
                    data[method].append(history[[metric]].dropna().min().values[0])
            fig, ax = plt.subplots(layout='constrained', figsize=(12, 5))
            x = np.arange(len(tasks))
            for method, values in data.items():
                width = 1 / len(values)
                offset = width * multiplier
                rects = ax.bar(x + offset, values, width=width, label=method)
                # ax.bar_label(rects, padding=3)
                multiplier += 1

            ax.set(xlabel='Task', ylabel=metric_to_ylabel_dict[metric], title=metric_to_title_dict[metric])
            ax.grid(False)
            ax.legend()
            ax.set_yscale('log')
            ax.set_xticks(x + width, [task_name_to_paper_name(task) for task in tasks])
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}_per_task.png'))
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}_per_task.svg'))
            plt.close(fig)

    for metric in ['val_rotation_mse_error', 
               'val_position_mse_error',
                'train_rotation_mse_error',
                'train_position_mse_error']:
        for type in histories.keys():
            tasks = histories[type].keys()
            data = {}
            exp_path = os.path.join(save_dir, type, 'barplots')
            os.makedirs(exp_path, exist_ok=True)
            multiplier = 0

            for task in histories[type].keys():
                for id, history in histories[type][task].items():
                    cfg = cfgs[type][task][id]
                    task_name = cfg['task']['name']
                    use_mask = cfg['policy'].get('use_mask', False)
                    method = target_to_method_name(cfg['policy']['_target_'], use_mask)
                    if method not in data:
                        data[method] = []
                    data[method].append(history[[metric]].dropna().min().values[0])
            
            data = {k: np.array(v).mean() for k, v in data.items()}
            fig, ax = plt.subplots(layout='constrained', figsize=(12, 5))
            x = np.arange(len(tasks))
            ax.bar(data.keys(), data.values())
            ax.set(xlabel='Task', ylabel=metric_to_ylabel_dict[metric], title=metric_to_title_dict[metric])
            ax.grid(False)
            ax.set_yscale('log')
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}.png'))
            plt.savefig(os.path.join(exp_path, f'{metric}_{type}.svg'))
            plt.close(fig)


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
