import numpy as np
from torch.autograd import grad
import theseus as th
from theseus.geometry.so3 import SO3
import torch

def marginal_prob_std(t, sigma=0.5):
    return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

def sample_from_se3_gaussian(x_mean, R_mean, std):
    x_eps = std[:,None]*torch.randn_like(x_mean)
    theta_eps = std[:,None]*torch.randn_like(x_mean)
    rot_eps = SO3().exp_map(theta_eps).to_matrix()

    _x = x_mean + x_eps
    _R = torch.einsum('bmn,bnk->bmk',R_mean, rot_eps)
    return _x, _R

def se3_log_probability_normal(x, R, x_tar, R_tar, std):

    ## Send to Theseus ##
    _R_tar = SO3()
    _R_tar.update(R_tar)

    if type(R) == torch.Tensor:
        _R = SO3()
        _R.update(R)
        R = _R


    ## Compute distance in R^3 + SO(3) ##
    R_tar_inv = _R_tar.inverse()
    dR = th.compose(R_tar_inv, R)
    dtheta = dR.log_map()

    dx = (x - x_tar)

    dist = torch.cat((dx, dtheta), dim=-1)
    return -.5*dist.pow(2).sum(-1)/(std.pow(2))

def se3_score_normal(x, R, x_tar, R_tar, std):
    if type(R) == torch.Tensor:
        _R = SO3()
        _R.update(R)
        R = _R

    theta = R.log_map()
    x_theta = torch.cat((x, theta), dim=-1)
    x_theta.requires_grad_(True)
    x = x_theta[..., :3]
    R = SO3.exp_map(x_theta[..., 3:])
    d = se3_log_probability_normal(x, R, x_tar, R_tar, std)
    v = grad(d.sum(), x_theta, only_inputs=True)[0]
    return v

def step(x, R, v):
    rot = SO3.exp_map(v[..., 3:]).to_matrix()
    R_1 = torch.einsum('bmn,bnk->bmk', rot, R)

    x_1 = x + v[...,:3]
    return x_1, R_1
