import torch
from diffusion_policy.common.rotation_utils import SO3_log_map, SO3_exp_map

def sample_xt(x0, x1, t):
    """
    Function which compute the sample xt along the geodesic from x0 to x1 on SE(3).
    """
    xt = x0*(1. - t[:,None]) + x1*t[:,None]
    return xt

def compute_conditional_vel(x1, xt, t):
    xt_dt = (x1 - xt)/ torch.clip((1 - t[:, None]), 0.01, 1.)
    return xt_dt

## Sample X at time t through the Geodesic from x0 -> x1
def sample_xt_SE3(x0, x1, t):
    """
    Function which compute the sample xt along the geodesic from x0 to x1 on SE(3).
    """
    ## Point to translation and rotation ##
    t_0, R_w_0 = x0[:, :3, -1], x0[:, :3, :3]
    t_1, R_w_1 = x1[:, :3, -1], x1[:, :3, :3]

    ## Get rot_t ##
    R_0_w = torch.transpose(R_w_0, -1, -2)
    R_0_1 = torch.matmul(R_0_w,R_w_1)
    lR_0_1 = SO3_log_map(R_0_1)
    R_0_t = SO3_exp_map(lR_0_1*t[:,None])
    R_w_t = torch.matmul(R_w_0, R_0_t)

    ## Get trans_t ##
    x_t = t_0*(1. - t[:,None]) + t_1*t[:,None]

    Ht = torch.eye(4, device=x0.device)[None,...].repeat(R_w_t.shape[0], 1, 1)
    Ht[:, :3, :3] = R_w_t
    Ht[:, :3, -1] = x_t
    return Ht

 ## Compute velocity target at xt at time t through the geodesic x0 -> x1
def compute_conditional_vel_SE3(H_w_1, H_w_t, t):
    x_t, R_w_t = H_w_t[:, :3, -1], H_w_t[:, :3, :3]
    x_1, R_w_1 = H_w_1[:, :3, -1], H_w_1[:, :3, :3]

    ## Compute Velocity in rot ##
    R_t_w = torch.transpose(R_w_t, -1, -2)
    R_t_1 = torch.matmul(R_t_w, R_w_1)
    lR_t_ut = SO3_log_map(R_t_1) / torch.clip((1 - t[:, None]), 0.01, 1.)

    ## Compute Velocity in trans ##
    x_ut = (x_1 - x_t)/ torch.clip((1 - t[:, None]), 0.01, 1.)

    return torch.cat((lR_t_ut, x_ut), dim=1).detach()

