#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --output=/home/urain/diffusion_policy/slurm_scripts/logs/%j/train.out
#SBATCH -J test_replay_policy

id=$(docker run -dt -v ${DIFFUSION_POLICY_ROOT}:/workspace 725f43c2daa2 2> /dev/null)
echo "Docker container id: $id"
echo "Starting test."
docker exec -t $id /bin/bash -c "source activate se3diffuser && 
                                pip3 install -e /workspace &&
                                cd /workspace/tests &&
                                xvfb-run -a python3 test_replay_keypoints.py"

echo "Finished."
