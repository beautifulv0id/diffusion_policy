"""
Usage:
Training:
python transformer_hybrid_obs_relative_encoder.py --config-name=transformer_hybrid_obs_relative_encoder
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.model.vision.transformer_hybrid_obs_relative_encoder import TransformerHybridObsRelativeEncoder

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config')),
    config_name='transformer_hybrid_obs_relative_encoder.yaml'
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    obs_encoder: TransformerHybridObsRelativeEncoder
    obs_encoder = hydra.utils.instantiate(cfg.obs_encoder)
    print("Output_shape: ", obs_encoder.output_shape())
    print("Visual output shape: ", obs_encoder.visual_output_shape())

if __name__ == "__main__":
    main()
