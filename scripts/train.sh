cd ${DIFFUSION_POLICY_ROOT}/diffusion_policy/workspace
. ~/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

args="num_episodes=1\
    training.rollout_every=1000\
    training.resume=True"

hydra_run_dir=$(python train_diffuser_actor.py $args training.init_resumable=True)
xvfb-run -a python train_diffuser_actor.py $args hydra.run.dir=$hydra_run_dir training.num_epochs=21 training.checkpoint_every=10
xvfb-run -a python train_diffuser_actor.py $args hydra.run.dir=$hydra_run_dir training.resume=True training.num_epochs=41 training.checkpoint_every=10

