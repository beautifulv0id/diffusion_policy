#!/bin/bash
training_script=$1 # "train_diffuser_actor.py"
SLURM_ARRAY_TASK_ID=$2
HYDRA_RUN_DIR_FILE=$3
NUM_GPUS=$4
args=${@:5}

echo "training_script: $training_script"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "args: $args"

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
echo "Running training script"
. ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/stud_herrmann/miniforge3/envs/se3diffuser
cd $DIFFUSION_POLICY_ROOT/diffusion_policy/workspace
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $NUM_GPUS $training_script $args hydra.run.dir=$hydra_run_dir

