import argparse
import os
import json
from diffusion_policy.common.json_logger import read_json_log
import wandb
api = wandb.Api()
entity = "felix-herrmann"
project = "diffusion_policy_debug"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--root", type=str, default="/home/felix/Workspace/diffusion_policy_felix/data/outputs")
args = arg_parser.parse_args()

def bad_run(path):
    valid = True
    if not os.path.exists(os.path.join(path, "checkpoints")):
        valid = False
    if not os.path.exists(os.path.join(path, "logs.json.txt")):
        valid = False
    else:
        log = read_json_log(os.path.join(path, "logs.json.txt"), required_keys="global_step")
        if "global_step" in log:
            if max(log["global_step"]) < 100000:
                valid = False
        else:
            valid = False
    return not valid

def delete_wandb_run(path):
    try:
        if os.path.exists(os.path.join(path, "wandb", "wandb-resume.json")):
            with open(os.path.join(path, "wandb", "wandb-resume.json"), "r") as f:
                run = json.load(f)['run_id']
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run}")
            run.delete()
            print(f"Deleted wandb run {run}")
        else:
            print(f"Could not find wandb resume file in {path}")
    except Exception as e:
        print(f"Could not delete wandb run: {e}")


def main():
    bad_runs = []
    for upper_dir in os.listdir(args.root):
        print(f"Checking {upper_dir}")
        for lower_dir in os.listdir(os.path.join(args.root, upper_dir)):
            # if no checkpoint folder
            path = os.path.join(args.root, upper_dir, lower_dir)
            # if single run
            if os.path.isdir(os.path.join(path, ".hydra")):
                if bad_run(path):
                    bad_runs.append(path)
                
            # check if its a multirun
            for folder in os.listdir(path):
                path = os.path.join(args.root, upper_dir, lower_dir)
                if str(folder).isdigit():
                    if bad_run(os.path.join(path, folder)):
                        bad_runs.append(os.path.join(path, folder))
        
    for run in bad_runs:
        print(f"Deleting {run}")
        delete_wandb_run(run)
        os.system(f"rm -rf {run}")
if __name__ == "__main__":
    main()