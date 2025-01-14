#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH -p gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --array=0-8%1
#SBATCH --output=../data/logs/%A_se3_flow_matching/train_%a.out
#SBATCH -J se3_flow_matching
#SBATCH --reservation=ias_41

config_name=train_se3_flow_matching_multi_gpu
tasks="[close_jar,open_drawer,push_buttons,reach_and_drag,sweep_to_dustpan_of_size,insert_onto_square_peg,place_cups,put_groceries_in_cupboard,slide_block_to_color_target,turn_tap,light_bulb_in,place_shape_in_shape_sorter,put_item_in_drawer,stack_blocks,meat_off_grill,place_wine_at_rack_location,put_money_in_safe,stack_cups]"
variations=$(seq -s, 0 199)
variations="[${variations}]"
job_name=se3_flow_matching
data_dir=${DIFFUSION_POLICY_ROOT}/data/diffuser_actor.zarr

args="tasks=${tasks}\
    variations=${variations}\
    optimizer.lr=4e-4\
    dataset.cache_size=1000\
    dataset.root=$data_dir\
    dataloader.batch_size=64\
    val_dataloader.batch_size=64\
    training.model_evaluation_every=10\
    training.checkpoint_every=1\
    training.save_milestone_every=100"


kwargs=${@:1}
    
args="$args $kwargs"

HYDRA_FULL_ERROR=1
HYDRA_RUN_DIR_FILE=${DIFFUSION_POLICY_ROOT}/data/logs/${SLURM_ARRAY_JOB_ID}_${job_name}/hydra_run_dir.txt
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/
. run_multi_gpu.sh $config_name \
            $SLURM_ARRAY_TASK_ID \
            $HYDRA_RUN_DIR_FILE \
            $args \
