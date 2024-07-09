#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -C 'rtx3090|a5000'
#SBATCH --output=/home/stud_herrmann/diffusion_policy_felix/.out/train/slurm-%j.out

task="open_drawer"
args="--save_path=${DIFFUSION_POLICY_ROOT}/data/images\
        --tasks=open_drawer\
        --episodes_per_task=50\
        --variations=1"

docker run -d --runtime=nvidia --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/felix/Workspace/diffusion_policy_felix/:/workspace --rm rlbench-diffuser
docker exec rlbench-diffuser bash -c "cd /installs/RlBench/tools && xvfb-run -a python dataset_generator.py $args"


