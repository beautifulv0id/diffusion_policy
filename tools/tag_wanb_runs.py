import os
import tap
import json
import wandb

class Arguments(tap.Tap):
    root_dir : str = os.path.join(os.environ['DIFFUSION_POLICY_ROOT'], 'data', 'outputs')
    project : str = 'diffusion_policy_debug'
    entity : str = 'felix-herrmann'
    tag : str
    remove_corrupted : bool = False

def main(args: Arguments):
    date_folders = os.listdir(args.root_dir)
    corrupted = {
        "dates": [],
        "experiments": []
    }
    api = wandb.Api()
    for date in date_folders:
        date_folder = os.path.join(args.root_dir, date)
        if not os.path.isdir(date_folder):
            corrupted["dates"].append((os.path.join(args.root_dir, date), "Not a directory"))
            continue
        for exp in os.listdir(date_folder):
            exp_folder = os.path.join(date_folder, exp)
            if not os.path.isdir(exp_folder):
                corrupted["experiments"].append((os.path.join(args.root_dir, exp_folder), "Not a directory"))
                continue
            wandb_resume_file = os.path.join(exp_folder, 'wandb', 'wandb-resume.json')
            if not os.path.isfile(wandb_resume_file):
                corrupted["experiments"].append(((os.path.join(args.root_dir, exp_folder), "wandb-resume.json not found")))
                continue
            with open(wandb_resume_file, 'r') as f:
                wandb_resume = json.load(f)
            run_id = wandb_resume['run_id']

            if not os.path.isfile(os.path.join(exp_folder, 'checkpoints', 'latest.ckpt')):
                corrupted["experiments"].append(((os.path.join(args.root_dir, exp_folder), "latest.ckpt not found")))
                continue

            try:
                run = api.run(f'{args.entity}/{args.project}/{run_id}')
            except Exception as e:
                corrupted["experiments"].append(((os.path.join(args.root_dir, exp_folder), f'Run not found: {e}')))
                continue
            run.tags.append(args.tag)
            run.update()
            print(f'{date}/{exp} -> {run_id}')

            

    if args.remove_corrupted:
        for k, v in corrupted.items():
            print(f'Corrupted {k}:')
            for e in v:
                print(f'\t{e[0]}: {e[1]}')
        print('Remove corrupted runs? [y/N]')
        if input().lower() == 'y':
            for k, v in corrupted.items():
                for e in v:
                    os.remove(e[0])

if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)