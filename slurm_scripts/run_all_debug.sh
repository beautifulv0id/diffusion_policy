args='training.debug=1 task.env_runner.max_episodes=1 task.env_runner.max_steps=1'

sbatch baseline_image/train_open_drawer.sh $args 
sbatch baseline_image/train_put_item_in_drawer.sh $args
sbatch baseline_image/train_stack_blocks.sh $args
sbatch baseline_image/train_sweep_to_dustpan_of_size.sh $args
sbatch baseline_image/train_turn_tap.sh $args
  
sbatch baseline_lowdim/train_open_drawer.sh $args
sbatch baseline_lowdim/train_put_item_in_drawer.sh $args
sbatch baseline_lowdim/train_stack_blocks.sh $args
sbatch baseline_lowdim/train_sweep_to_dustpan_of_size.sh $args
sbatch baseline_lowdim/train_turn_tap.sh $args

sbatch baseline_mask/train_open_drawer.sh $args
sbatch baseline_mask/train_put_item_in_drawer.sh $args
sbatch baseline_mask/train_stack_blocks.sh $args
sbatch baseline_mask/train_sweep_to_dustpan_of_size.sh $args
sbatch baseline_mask/train_turn_tap.sh $args
  
sbatch action_flow_image/train_open_drawer.sh $args
sbatch action_flow_image/train_put_item_in_drawer.sh $args
sbatch action_flow_image/train_stack_blocks.sh $args
sbatch action_flow_image/train_sweep_to_dustpan_of_size.sh $args
sbatch action_flow_image/train_turn_tap.sh $args
  
sbatch action_flow_lowdim/train_open_drawer.sh $args
sbatch action_flow_lowdim/train_put_item_in_drawer.sh $args
sbatch action_flow_lowdim/train_stack_blocks.sh $args
sbatch action_flow_lowdim/train_sweep_to_dustpan_of_size.sh $args
sbatch action_flow_lowdim/train_turn_tap.sh $args
  
sbatch action_flow_mask/train_open_drawer.sh $args
sbatch action_flow_mask/train_put_item_in_drawer.sh $args
sbatch action_flow_mask/train_stack_blocks.sh $args
sbatch action_flow_mask/train_sweep_to_dustpan_of_size.sh $args
sbatch action_flow_mask/train_turn_tap.sh $args