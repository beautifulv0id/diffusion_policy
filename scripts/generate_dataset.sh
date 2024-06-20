save_path=/home/felix/Workspace/diffusion_policy_felix/data/image
image_size=256,256
variations=1
episodes_per_task=100
all_variations=False
processes=1
tasks=open_drawer
high_dim=True

cd /home/felix/Workspace/diffusion_policy_felix/installs/RLBench/tools
xvfb-run -a /home/felix/miniforge3/envs/robodiff/bin/python3 dataset_generator.py \
--tasks=$tasks \
--all_variations=False \
--save_path=$save_path \
--variations=$variations \
--episodes_per_task=$episodes_per_task \
--image_size=$image_size \
--processes=$processes \
--high_dim=$high_dim
