
from_server=stud_zach@mn.ias.informatik.tu-darmstadt.de
from_root=/home/stud_zach/diffusion_policy_felix
from_path=data/outputs/2024.07.16/19.27.52_train_diffuser_actor_stack_blocks_mask/

root=$DIFFUSION_POLICY_ROOT
this_path=$root/$from_path

mkdir -p $this_path
mkdir -p $this_path/checkpoints
mkdir -p $this_path/.hydra

scp -r $from_server:$from_root/$from_path/checkpoints/ $this_path
scp -r $from_server:$from_root/$from_path/.hydra/ $this_path
