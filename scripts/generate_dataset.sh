save_path=${DIFFUSION_POLICY_ROOT}/data/image4
image_size=128,128
variations=1
episodes_per_task=3
processes=3
tasks=sweep_to_dustpan_of_size
high_dim=True

cd ${DIFFUSION_POLICY_ROOT}/tools
xvfb-run -a /home/felix/miniforge3/envs/robodiff/bin/python3 dataset_generator.py \
    --tasks=$tasks \
    --save_path=$save_path \
    --variations=$variations \
    --episodes_per_task=$episodes_per_task \
    --image_size=$image_size \
    --processes=$processes \
    --high_dim=$high_dim
