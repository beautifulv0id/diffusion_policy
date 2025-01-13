#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%1
#SBATCH --output=../data/logs/%A_action_flow/train_%a.out
#SBATCH -J action_flow

training_script=train_action_flow.py
tasks="[close_jar,open_drawer,push_buttons,reach_and_drag,sweep_to_dustpan_of_size,insert_onto_square_peg,place_cups,put_groceries_in_cupboard,slide_block_to_color_target,turn_tap,light_bulb_in,place_shape_in_shape_sorter,put_item_in_drawer,stack_blocks,meat_off_grill,place_wine_at_rack_location,put_money_in_safe,stack_cups]"
job_name=action_flow
data_dir=${DIFFUSION_POLICY_ROOT}/data/diffuser_actor.zarr

args="tasks=${tasks}\
    training.resume=True\
    env_runner.n_procs_max=5\
    dataset.use_precomputed_features=False\
    dataset.root=$data_dir\
    dataset.image_rescale=[0.75,1.25]"

if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]; then
    args="$args mode=rollout"
fi

kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_RUN_DIR_FILE=${DIFFUSION_POLICY_ROOT}/data/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run.sh $training_script \
            $SLURM_ARRAY_TASK_ID \
            $HYDRA_RUN_DIR_FILE \
            $args \
