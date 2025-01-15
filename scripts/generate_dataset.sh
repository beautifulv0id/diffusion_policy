#save_path=${DIFFUSION_POLICY_ROOT}/data/rlbench
save_path=/media/funk/INTENSO/300_RL_BENCH/data_20_12/rlbench
image_size=128,128
variations=1
episodes_per_task=3
processes=5
#tasks=open_drawer,put_item_in_drawer,stack_blocks,sweep_to_dustpan_of_size,turn_tap,slide_block_to_target,meat_off_grill,close_jar,screw_nail,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups
tasks=turn_tap,slide_block_to_target,meat_off_grill,close_jar,screw_nail,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups
high_dim=true

cd ${DIFFUSION_POLICY_ROOT}/tools

tasks=turn_tap,slide_block_to_target,meat_off_grill,close_jar,screw_nail,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups
python dataset_generator.py --tasks=$tasks --save_path=$save_path'/train' --variations=$variations --episodes_per_task=200 --image_size=$image_size --processes=$processes --high_dim=$high_dim
tasks=open_drawer,put_item_in_drawer,stack_blocks,sweep_to_dustpan_of_size,turn_tap,slide_block_to_target,meat_off_grill,close_jar,screw_nail,put_money_in_safe,stack_wine,put_groceries_in_cupboard,place_shape_in_shape_sorter,push_buttons,insert_onto_square_peg,stack_cups,place_cups
python dataset_generator.py --tasks=$tasks --save_path=$save_path'/val' --variations=$variations --episodes_per_task=50 --image_size=$image_size --processes=$processes --high_dim=$high_dim
python dataset_generator.py --tasks=$tasks --save_path=$save_path'/test' --variations=$variations --episodes_per_task=50 --image_size=$image_size --processes=$processes --high_dim=$high_dim
