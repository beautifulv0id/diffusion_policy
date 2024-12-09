
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/baseline_image
sbatch train_open_drawer.sh 
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap.sh

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/baseline_lowdim
sbatch train_open_drawer.sh
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap.sh
  
cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/baseline_mask
sbatch train_open_drawer.sh
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap.sh

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/action_flow_image
sbatch train_open_drawer.sh
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap.sh

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/action_flow_lowdim
sbatch train_open_drawer.sh
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap_image.sh

cd ${DIFFUSION_POLICY_ROOT}/slurm_scripts/action_flow_mask
sbatch train_open_drawer.sh
sbatch train_put_item_in_drawer.sh
sbatch train_stack_blocks.sh
sbatch train_sweep_to_dustpan_of_size.sh
sbatch train_turn_tap.sh