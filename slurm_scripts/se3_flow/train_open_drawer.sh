#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=../../data/logs/%j_open_drawer_image/out

. ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/stud_herrmann/miniforge3/envs/se3diffuser
cd $DIFFUSION_POLICY_ROOT/diffusion_policy/workspace

export HYDRA_FULL_ERROR=1
python train_se3_flow_matching.py task.dataset.use_precomputed_features=True dataloader.batch_size=16 val_dataloader.batch_size=4