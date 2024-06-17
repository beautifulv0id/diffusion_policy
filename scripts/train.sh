cd /home/felix/Workspace/diffusion_policy_felix/diffusion_policy/workspace/
conda activate robodiff

python train_flow_matching_SE3_lowdim_workspace.py \
                                        task=put_item_in_drawer \
                                        num_episodes=1 \
                                        training.rollout_every=5000 \
                                        training.num_epochs=50000 \
                                        +task.env_runner.n_train_vis=3 \
                                        +task.env_runner.n_val_vis=3 \
                                        task.env_runner.max_episodes=5 \
                                        policy.relative_position=False \
                                        policy.relative_rotation=False \
                                        policy.data_augmentation=False 

# python train_flow_matching_SE3_lowdim_workspace.py \
#                                         task=put_item_in_drawer \
#                                         num_episodes=1 \
#                                         training.rollout_every=5000 \
#                                         training.num_epochs=50000 \
#                                         +task.env_runner.n_train_vis=3 \
#                                         +task.env_runner.n_val_vis=3 \
#                                         task.env_runner.max_episodes=5 \
#                                         policy.relative_position=True \
#                                         policy.relative_rotation=False \
#                                         policy.data_augmentation=True 

# python train_flow_matching_SE3_lowdim_workspace.py \
#                                         task=open_drawer \
#                                         num_episodes=-1 \
#                                         training.rollout_every=150000 \
#                                         training.num_epochs=150000 \
#                                         +task.env_runner.n_train_vis=3 \
#                                         +task.env_runner.n_val_vis=3 \
#                                         task.env_runner.max_episodes=5 \
#                                         policy.relative_position=False \
#                                         policy.relative_rotation=False \
#                                         policy.data_augmentation=True 