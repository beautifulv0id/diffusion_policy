#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -p gpu
#SBATCH --output=/home/urain/diffusion_policy/.out/log-%j.out
#SBATCH -J rlbench_to_zarr

cd /home/urain/diffusion_policy/data_preprocessing
. ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/stud_herrmann/miniforge3/envs/se3diffuser

python rlbench_to_zarr.py --save_path /home/urain/diffusion_policy/data/image.zarr \
        --data_path /home/urain/diffusion_policy/data/image \
        --n_demos -1