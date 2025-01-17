"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import copy
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch
import os
import random
import numpy as np

def initialize_distributed(seed):
    is_torchrun = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_torchrun:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # DDP initialization
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')),
    config_name='train_se3_flow_matching.yaml'
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    cfg_unresolved = copy.deepcopy(cfg)
    OmegaConf.resolve(cfg)
    initialize_distributed(cfg.training.seed)
    cls = hydra.utils.get_class(cfg._target_)
    # Idea: pass also the unresolved config as this later makes the evaluation across different devices significantly more convenient
    # since all the paths are passed as env variables if done correctly!
    workspace: BaseWorkspace = cls(cfg, cfg_unresolved=cfg_unresolved)
    workspace.run()

if __name__ == "__main__":
    main()
