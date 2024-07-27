args='training.debug=1 task.env_runner.max_episodes=1 task.env_runner.max_steps=1'

sbatch baseline_image/train_open_drawer_image.sh $args 
sbatch baseline_image/train_put_item_in_drawer_image.sh $args
sbatch baseline_image/train_stack_blocks_image.sh $args
sbatch baseline_image/train_sweep_to_dustpan_of_size_image.sh $args
sbatch baseline_image/train_turn_tap_image.sh $args

  
sbatch baseline_lowdim/train_open_drawer_lowdim.sh $args
sbatch baseline_lowdim/train_put_item_in_drawer_lowdim.sh $args
sbatch baseline_lowdim/train_stack_blocks_lowdim.sh $args
sbatch baseline_lowdim/train_sweep_to_dustpan_of_size_lowdim.sh $args
sbatch baseline_lowdim/train_turn_tap_lowdim.sh $args


  
sbatch baseline_mask/train_open_drawer_mask.sh $args
sbatch baseline_mask/train_put_item_in_drawer_mask.sh $args
sbatch baseline_mask/train_stack_blocks_mask.sh $args
sbatch baseline_mask/train_sweep_to_dustpan_of_size_mask.sh $args
sbatch baseline_mask/train_turn_tap_mask.sh $args


sbatch pose_invariant_image/train_open_drawer_image_v2.sh $args
sbatch pose_invariant_image/train_put_item_in_drawer_image_v2.sh $args
sbatch pose_invariant_image/train_stack_blocks_image_v2.sh $args
sbatch pose_invariant_image/train_sweep_to_dustpan_of_size_image_v2.sh $args
sbatch pose_invariant_image/train_turn_tap_image_v2.sh $args

  
sbatch pose_invariant_lowdim/train_open_drawer_lowdim.sh $args
sbatch pose_invariant_lowdim/train_put_item_in_drawer_lowdim.sh $args
sbatch pose_invariant_lowdim/train_stack_blocks_lowdim.sh $args
sbatch pose_invariant_lowdim/train_sweep_to_dustpan_of_size_lowdim.sh $args
sbatch pose_invariant_lowdim/train_turn_tap_image_lowdim.sh $args

  
sbatch pose_invariant_lowdim_v2/train_open_drawer_lowdim_v2.sh $args
sbatch pose_invariant_lowdim_v2/train_put_item_in_drawer_lowdim_v2.sh $args
sbatch pose_invariant_lowdim_v2/train_stack_blocks_lowdim_v2.sh $args
sbatch pose_invariant_lowdim_v2/train_sweep_to_dustpan_of_size_lowdim_v2.sh $args
sbatch pose_invariant_lowdim_v2/train_turn_tap_image_lowdim_v2.sh $args

  
sbatch pose_invariant_mask/train_open_drawer_mask_v2.sh $args
sbatch pose_invariant_mask/train_put_item_in_drawer_mask_v2.sh $args
sbatch pose_invariant_mask/train_stack_blocks_mask_v2.sh $args
sbatch pose_invariant_mask/train_sweep_to_dustpan_of_size_mask_v2.sh $args
sbatch pose_invariant_mask/train_turn_tap_mask_v2.sh $args