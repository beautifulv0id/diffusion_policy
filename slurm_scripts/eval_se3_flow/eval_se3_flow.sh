#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 15
#SBATCH --mem=15G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000|a6000'

training_script=eval_se3_flow_matching_post_train_test.py

args="model_to_eval_path=TODO"

kwargs=${@:1}
    
args="$args $kwargs"

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/

id=$(docker run -e WANDB_API_KEY=$WANDB_API_KEY -e DIFFUSION_POLICY_DATA_ROOT=$DIFFUSION_POLICY_DATA_ROOT -dt  -v ${DIFFUSION_POLICY_ROOT}:/workspace oddtoddler400/pointattention:latest)
echo "Container ID: $id"
echo "Running training script"
docker exec -t $id /bin/bash -c "source activate se3diffuser &&
                        cd /workspace/diffusion_policy/workspace &&
                        HYDRA_FULL_ERROR=1 xvfb-run -a python3 $training_script $args"
docker stop $id
