#!/bin/bash
config_name=$1 # "train_diffuser_actor.py"
SLURM_ARRAY_TASK_ID=$2
HYDRA_RUN_DIR_FILE=$3
args=${@:4}

echo "config: $config_name"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "args: $args"

check_and_make_hydra_run_dir() {
    for arg in "$@"; do
        if [[ "$arg" == hydra.run.dir=* ]]; then
            hydra_run_dir=${arg#hydra.run.dir=}
            echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
            return 1
        fi
    done
    return 0
}

check_and_make_hydra_run_dir $args

if [ $SLURM_ARRAY_TASK_ID -eq 0 ] && [ ! -f $HYDRA_RUN_DIR_FILE ];
then
    hydra_run_dir=$(./get_hydra_path.sh $config_name)
    echo $hydra_run_dir > $HYDRA_RUN_DIR_FILE
fi

hydra_run_dir=$(cat $HYDRA_RUN_DIR_FILE)
id=$(docker run -e WANDB_API_KEY=$WANDB_API_KEY -dt  -v ${DIFFUSION_POLICY_ROOT}:/workspace -v /home/share/3D_attn_felix:/docker_data_dir -v /home/funk/Code/3d_repr/pointattention:/pt_attn oddtoddler400/pointattention:latest)
echo "Container ID: $id"
echo "Running training script"
docker exec -t $id /bin/bash -c "source activate se3diffuser &&
                        cd /pt_attn/ &&
                        pip uninstall -y geo3dattn &&
                        pip install -e . &&
                        cd /workspace/ &&
                        HYDRA_FULL_ERROR=1 xvfb-run python train.py --config-name $config_name hydra.run.dir=$hydra_run_dir $args"
docker stop $id

