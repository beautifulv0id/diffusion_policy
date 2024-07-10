#!/bin/bash
training_script=$1 # "train_diffuser_actor.py"
task_name=$2 #" put_item_in_drawer"
task_config=$3 # "put_item_in_drawer_mask"
SLURM_ARRAY_TASK_ID=$4
args=${@:5}

echo "training_script: $training_script"
echo "task_name: $task_name"
echo "task_config: $task_config"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "args: $args"

WANDB_API_KEY=8009cee998358d908f42c2fce77f1ee094836701
HYDRA_RUN_DIR_FILE=/home/stud_herrmann/diffusion_policy_felix/slurm_scripts/logs/${SLURM_ARRAY_JOB_ID}_${task_config}/hydra_run_dir.txt

check_and_make_hydra_run_dir() {
    for arg in "$@"; do
        if [[ "$arg" == hydra.run.dir=* ]]; then
            hydra_run_dir=${arg#hydra.run.dir=}
            echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
            return 1
        fi
    done
    return 0
}

check_and_make_hydra_run_dir $args

if [ $SLURM_ARRAY_TASK_ID -eq 0 ] && [ ! -f $HYDRA_RUN_DIR_FILE ];
then
    hydra_run_dir=$(./get_hydra_path.sh $training_script $task_config)
    echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
fi

hydra_run_dir=$(cat $HYDRA_RUN_DIR_FILE)
id=$(docker run -dt  -v ${DIFFUSION_POLICY_ROOT}:/workspace 725f43c2daa2)
echo "Container ID: $id"
echo "Setting up python environment"
docker exec -t $id /bin/bash -c "cd /workspace/installs/RLBench && python setup.py develop &&
                        cd /workspace/installs/PyRep && pip install . &&
                        cd /workspace && pip install -e ."
echo "Running training script"
docker exec -t -e DGLBACKEND=pytorch -e WANDB_API_KEY=$WANDB_API_KEY $id /bin/bash -c "source activate se3diffuser && 
                        cd /workspace/diffusion_policy/workspace &&
                        xvfb-run -a python3 $training_script $args hydra.run.dir=$hydra_run_dir"
docker stop $id

