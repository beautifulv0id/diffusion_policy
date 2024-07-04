#!/bin/bash
# This is test.bash
#SBATCH -t 00:30:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G
#SBATCH -p stud3080,stud,dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/train_%j.out


training_script="train_diffuser_actor.py"

task_name="put_item_in_drawer"
task_config="put_item_in_drawer_mask"

args="task=$task_config\
    num_episodes=-1\
    training.resume=True"

WANDB_API_KEY=8009cee998358d908f42c2fce77f1ee094836701
CONTAINER_ID_FILE=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/train_${SLURM_ARRAY_JOB_ID}_container_id.txt
HYDRA_RUN_DIR_FILE=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/train_${SLURM_ARRAY_JOB_ID}_hydra_run_dir.txt

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]
then
    echo "Starting docker container."
    id=$(docker run -dt --shm-size=8g -v ${DIFFUSION_POLICY_ROOT}:/workspace 725f43c2daa2 2> /dev/null)
    hydra_run_dir=$(./get_hydra_path.sh $training_script $task_name)
    echo $id > $CONTAINER_ID_FILE
    echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
fi

echo "Starting training."
id=$(cat $CONTAINER_ID_FILE)
hydra_run_dir=$(cat $HYDRA_RUN_DIR_FILE)
echo "Docker container id: $id"
echo "Hydra run directory: $hydra_run_dir"
echo "Arguments: $args"
docker exec $id /bin/bash -c "source activate se3diffuser && 
                        export DGLBACKEND=pytorch &&
                        export WANDB_API_KEY=$WANDB_API_KEY &&
                        cd /workspace/diffusion_policy/workspace && 
                        xvfb-run -a python3 $training_script $args hydra.run.dir=$hydra_run_dir"

