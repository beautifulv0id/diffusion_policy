#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --array=0-3%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/%A/train_%a.out


training_script=$1 # "train_diffuser_actor.py"
task_name=$2 #" put_item_in_drawer"
task_config=$3 # "put_item_in_drawer_mask"

args="task=$task_config\
    num_episodes=-1\
    training.resume=True\
    dataloader.batch_size=48\
    val_dataloader.batch_size=48"

WANDB_API_KEY=8009cee998358d908f42c2fce77f1ee094836701
HYDRA_RUN_DIR_FILE=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/${SLURM_ARRAY_JOB_ID}/hydra_run_dir.txt

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]
then
    hydra_run_dir=$(./get_hydra_path.sh $training_script $task_name)
    echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
fi

hydra_run_dir=$(cat $HYDRA_RUN_DIR_FILE)
docker run -t -e DGLBACKEND=pytorch -e WANDB_API_KEY=$WANDB_API_KEY  -v ${DIFFUSION_POLICY_ROOT}:/workspace 725f43c2daa2 /bin/bash -c "source activate se3diffuser && cd /workspace/diffusion_policy/workspace && xvfb-run -a python3 $training_script $args hydra.run.dir=$hydra_run_dir"
