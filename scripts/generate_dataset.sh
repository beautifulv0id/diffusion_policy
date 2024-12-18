save_path=${DIFFUSION_POLICY_ROOT}/data/rlbench
image_size=128,128
variations=1
episodes_per_task=3
processes=5
tasks=put_item_in_drawer,stack_blocks,sweep_to_dustpan_of_size,turn_tap,slide_block_to_target,meat_off_grill,close_jar,screw_nail,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups
high_dim=true

cd ${DIFFUSION_POLICY_ROOT}/tools

python dataset_generator.py --tasks=$tasks --save_path=$save_path'/train' --variations=$variations --episodes_per_task=100 --image_size=$image_size --processes=$processes --high_dim=$high_dim
python dataset_generator.py --tasks=$tasks --save_path=$save_path'/val' --variations=$variations --episodes_per_task=25 --image_size=$image_size --processes=$processes --high_dim=$high_dim
python dataset_generator.py --tasks=$tasks --save_path=$save_path'/test' --variations=$variations --episodes_per_task=25 --image_size=$image_size --processes=$processes --high_dim=$high_dim

