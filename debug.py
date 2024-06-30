from diffusion_policy.env.rlbench.rlbench_env import get_masks

import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()

if __name__ == '__main__':
    print("Starting")
    get_masks()
