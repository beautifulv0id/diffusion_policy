from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import einsum, matmul
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from pytorch3d.transforms import matrix_to_quaternion, standardize_quaternion
from diffusion_policy.model.common.so3_util import log_map, normal_so3
from diffusion_policy.model.flow_matching.flow_matching_models import RectifiedLinearFlow, LinearAttractorFlow, SE3LinearAttractorFlow

'''
A policy class that represents the action space as a trajectory of SE(3) poses T=(R,p), with R the rotation matrix and p the translation vector.
The policy is trained to predict the action trajectory given a sequence of observations.
The policy is trained as a Continuous Normalizing Flow using Conditional Flow Matching in SO(3) and R^3. 
'''
class SE3FlowMatchingPolicy(BaseLowdimPolicy):
    def __init__(self,
                 model,
                 obs_encoder,
                 horizon,
                 n_obs_steps,
                 num_inference_steps=10,
                 flow_type='se3_linear_attractor',#'linear_attractor', 'rectified_linear', 'se3_linear_attractor'
                 t_switch=.75,
                 gripper_out = False,
                 ignore_collisions_out=False,
                 translation_loss_scaling=1.,
                 rotation_loss_scaling=1.,
                 only_inference_time=False,
                 normalizer=None,
                 batch_adaptor=None,
                 augmentor=None,
                 **kwargs):
        super(SE3FlowMatchingPolicy, self).__init__()

        ### MODEL ####
        self.model = model
        self.obs_encoder = obs_encoder

        ## Normalizer ##
        self.normalizer = normalizer

        ## Batch Adaptor ##
        self.batch_adaptor = batch_adaptor

        ## Augmentor ##
        self.augmentor = augmentor

        ## Parameters ##
        self.only_inference_time = only_inference_time
        self.translation_loss_scaling = translation_loss_scaling
        self.rotation_loss_scaling = rotation_loss_scaling
        self.action_dim = (4,4)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs
        self._output_target = False

        ## Inference ##
        self.num_inference_steps = num_inference_steps

        ## Flow Model ##
        self.t_switch = t_switch
        if flow_type == 'se3_linear_attractor':
            self.flow = SE3LinearAttractorFlow(t_switch=t_switch)
        elif flow_type == 'linear_attractor':
            self.flow = LinearAttractorFlow(t_switch=t_switch)
        else:
            self.flow = RectifiedLinearFlow(world_translation=True)
        self.generate_random_initial_pose = self.flow.generate_random_initial_pose
        self.flow_at_t = self.flow.flow_at_t
        self.vector_field_at_t = self.flow.vector_field_at_t
        self.step = self.flow.step

        ## Gripper Actor ##
        self.gripper_out = gripper_out
        self.ignore_collisions_out = ignore_collisions_out

        self.set_flow_matching_normalization_params()

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], batch=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if self.normalizer is not None:
            obs_dict = self.normalizer.normalize({'obs': obs_dict})['obs']

        ## Set Observations as Context ##
        if self.batch_adaptor is not None:
            obs_dict = self.batch_adaptor.adapt({'obs': obs_dict})['obs']
        latent_obs = self.obs_encoder(obs_dict)
        self.model.set_context(latent_obs)

        ## run sampling
        if batch is None:
            batch = obs_dict['robot0_eef_pos'].shape[0]

        sample = self.sample(
            batch_size=batch
        )

        # unnormalize prediction
        action_pred = sample['action']
        extra = sample.get('extra', {})

        result = {
            'action_type':'absolute_pose',
            'action': action_pred,
            'obs': obs_dict,
            'extra': extra
        }

        if self.normalizer is not None:
            result = self.normalizer.unnormalize(result)


        return result

    # ========= training  ============   
    def set_flow_matching_normalization_params(self):
        self.ff_position_scale = 1.

    def compute_loss(self, batch):
        if self.augmentor is not None:
            batch = self.augmentor.augment(batch)

        if self.normalizer is not None:
            batch = self.normalizer.normalize(batch)
        batch_size, act_steps = batch['action']['act_p'].shape[0], batch['action']['act_p'].shape[1]
        device, dtype =  batch['action']['act_p'].device,  batch['action']['act_p'].dtype
        ##  Input ##
        obs = batch['obs']
        r1, p1 = batch['action']['act_r'], batch['action']['act_p']
        r0, p0 = self.generate_random_initial_pose(batch_size, act_steps, position_scale=self.ff_position_scale)
        r0, p0 = r0.to(device), p0.to(device)

        # Sample noise that we'll add to the actions
        if self.only_inference_time:
            time = torch.randint(0, self.num_inference_steps, (batch_size,), device=device)/self.num_inference_steps
        else:
            time = torch.rand(batch_size, device=device, dtype=dtype)

        ## Sample a at time t given the Flow \Phi_t ##
        rt, pt = self.flow_at_t(r0, p0, r1, p1, time)
        ## Compute Target Velocity  given  u_t = d\Phi_t/dt##
        d_rt, d_pt = self.vector_field_at_t(r1,p1,rt,pt,time)

        # Predict velocity for at
        ## Compute Latent Observation ##

        if self.batch_adaptor is not None:
            obs = self.batch_adaptor.adapt({"obs": obs})["obs"]
        latent_obs = self.obs_encoder(obs)

        self.model.set_context(latent_obs)
        model_out = self.model(rt, pt, time*self.num_inference_steps)
        d_act = model_out['v']
        dr_pred, dp_pred = d_act[...,:3], d_act[...,3:6]

        loss = self.rotation_loss_scaling*F.mse_loss(dr_pred, d_rt, reduction='none') + self.translation_loss_scaling*F.mse_loss(dp_pred, d_pt, reduction='none')/self.ff_position_scale
        loss = loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        ## Add Gripper Loss ##
        if self.gripper_out:
            pred_action = model_out['gripper']
            target_action = batch['action']['act_gr']
            bce_loss = F.binary_cross_entropy(pred_action, target_action, reduction='none')
            bce_loss = reduce(bce_loss, 'b ... -> b (...)', 'mean')
            loss += bce_loss.mean()

        ## Add Ignore Collision Loss ##
        if self.ignore_collisions_out:
            pred_action = model_out['ignore_collisions']
            target_action = batch['action']['act_ic']
            bce_loss = F.binary_cross_entropy(pred_action, target_action, reduction='none')
            bce_loss = reduce(bce_loss, 'b ... -> b (...)', 'mean')
            loss += bce_loss.mean()

        return loss

    def evaluate(self, batch):
        log_dict = {}

        gt_action = batch['action']
        gt_act_p = gt_action['act_p']
        gt_act_r = gt_action['act_r']

        out = self.predict_action(batch['obs'])
        action = out['action']
        pred_act_p = action['act_p']
        pred_act_r = action['act_r']
        if self.gripper_out:
            pred_act_gr = out['extra']['act_gr_pred']
        if self.ignore_collisions_out:
            pred_act_ic = out['extra']['act_ic_pred']

        pos_error = torch.nn.functional.mse_loss(pred_act_p, gt_act_p)

        R_inv_gt = torch.transpose(gt_act_r, -1, -2)
        relative_R = torch.matmul(R_inv_gt, pred_act_r)
        angle_error = log_map(relative_R)
        rot_error = torch.nn.functional.mse_loss(angle_error, torch.zeros_like(angle_error))

        if self.gripper_out:
            gt_act_gr = gt_action['act_gr']
            gr_error = torch.nn.functional.mse_loss(pred_act_gr, gt_act_gr)
            log_dict['train_gripper_mse_error'] = gr_error.item()

        if self.ignore_collisions_out:
            gt_act_ic = gt_action['act_ic']
            ic_error = torch.nn.functional.mse_loss(pred_act_ic, gt_act_ic)
            log_dict['train_ignore_collisions_mse_error'] = ic_error.item()

        log_dict['train_position_mse_error'] = pos_error.item()
        log_dict['train_rotation_mse_error'] = rot_error.item()

        return log_dict

    # ========= Utils ============

    def sample(self, batch_size):

        with torch.no_grad():
            dt = 1 / self.num_inference_steps

            # sample H_0
            r0, p0 = self.generate_random_initial_pose(batch_size, self.horizon)
            r0, p0 = r0.to(self.device), p0.to(self.device)

            rt, pt = r0, p0
            for s in range(0, self.num_inference_steps):
                time = s*dt*torch.ones_like(pt[:, 0, 0])
                model_out = self.model(rt, pt, time*self.num_inference_steps)
                d_act = model_out['v']
                dr_pred, dp_pred = d_act[...,:3], d_act[...,3:6]
                rt, pt = self.step(rt, pt, dr_pred, dp_pred, dt, time=s*dt, output_target=self._output_target)

            out_dict = {'action': {'act_r': rt, 'act_p': pt}, "extra": {}}
            if self.gripper_out:
                act_gr = model_out['gripper'] > 0.5
                out_dict['extra'] ['act_gr_pred'] = act_gr
                out_dict['action']['act_gr'] = act_gr
            if self.ignore_collisions_out:
                act_ic = model_out['ignore_collisions'] > 0.5
                out_dict['extra']['act_ic_pred'] = act_ic
                out_dict['action']['act_ic'] = act_ic

            return out_dict

    def get_args(self):
        args_dict = {
            '__class__': [type(self).__module__, type(self).__name__],
            'pre_load':{
                'model': self.model.get_args(),
                'obs_encoder':self.obs_encoder.get_args(),},
            'params':{
                'horizon': self.horizon,
                'n_obs_steps': self.n_obs_steps,
                'num_inference_steps': self.num_inference_steps,
                't_switch': self.t_switch,
            },
        }
        return args_dict




############# TEST ################
def load_model(horizon = 12, n_obs_steps=2, num_inference_steps = 10, dim_features = 50, device='cpu', obs_dim=50):
    from diffusion_policy.model.decoder.pose_git import FlowMatchingInvariantPointTransformer
    from diffusion_policy.model.obs_encoders.base_model import BaseObservationEncoder


    horizon = horizon
    n_obs_steps = n_obs_steps

    t_switch = 0.75
    num_inference_steps = num_inference_steps

    ## Load the Invariant Transformer as Policy##
    latent_dim = 128
    depth = 2
    heads = 3
    dim_head = 60
    model = FlowMatchingInvariantPointTransformer(obs_dim=obs_dim,
                                                n_obs_steps=n_obs_steps,
                                                  horizon=horizon,
                                                  latent_dim=latent_dim,
                                                  depth=depth,
                                                  heads=heads,
                                                  dim_head=dim_head,
                                                  dropout=0.,).to(device)

    ## Load the Observation Encoder ##
    obs_encoder = BaseObservationEncoder()

    ## Load SE3FlowMatchingPolicy ##
    policy = SE3FlowMatchingPolicy(model=model,
                               obs_encoder=obs_encoder,
                               horizon=horizon,
                               n_obs_steps=n_obs_steps,
                               num_inference_steps=num_inference_steps,
                               t_switch=t_switch,
                               only_inference_time=True)
    return policy


def test_flow_and_vector_field():
    import matplotlib.pyplot as plt

    horizon = 12
    num_inference_steps = 4
    policy = load_model(horizon=horizon, num_inference_steps=num_inference_steps)

    #################### FLOW ########################
    time_steps = num_inference_steps + 1
    r1 = torch.eye(3)[None, None, ...].repeat(time_steps, horizon, 1, 1)
    p1 = torch.zeros(time_steps, horizon, 3)

    r0, p0 = policy.generate_random_initial_pose(batch=1, trj_steps=horizon)
    r0, p0 = r0[:, ...].repeat(time_steps, 1, 1, 1), p0[:, ...].repeat(time_steps, 1, 1)

    time = torch.linspace(0, 1, time_steps)
    rt, pt = policy.flow_at_t(r0, p0, r1, p1, time)

    
    fig, ax = plt.subplots(nrows=3, ncols=2)

    vt = log_map(rt)
    for i in range(horizon):
        ax[0,0].plot(pt[:, i, 0], 'r')
        ax[1,0].plot(pt[:, i, 1], 'r')
        ax[2,0].plot(pt[:, i, 2], 'r')

        ax[0,1].plot(vt[:, i, 0], 'r')
        ax[1,1].plot(vt[:, i, 1], 'r')
        ax[2,1].plot(vt[:, i, 2], 'r')

    #################### VECTOR FIELD ########################
    _rt, _pt = r0[0, ...], p0[0, ...]
    _r1, _p1 = r1[0, ...], p1[0, ...]
    trj_r, trj_p = _rt[None, ...], _pt[None, ...]

    dt = 1 / num_inference_steps
    for t in range(0, num_inference_steps):
        d_rt, d_pt = policy.vector_field_at_t(_r1[None, ...], _p1[None, ...], _rt[None, ...], _pt[None, ...],
                                              dt * t * torch.ones_like(_p1[:1, 0]))
        _rt, _pt = policy.step(_rt, _pt, d_rt[0, ...], d_pt[0, ...], dt, time=t*dt)

        ## Save in Trajectory ##
        trj_r, trj_p = torch.cat((trj_r, _rt[None, ...]), dim=0), torch.cat((trj_p, _pt[None, ...]), dim=0)

    ## Visualize ##
    vt2 = log_map(trj_r)
    pt2 = trj_p
    for i in range(horizon):
        ax[0,0].plot(pt2[:, i, 0], 'b')
        ax[1,0].plot(pt2[:, i, 1], 'b')
        ax[2,0].plot(pt2[:, i, 2], 'b')

        ax[0,1].plot(vt2[:, i, 0], 'b')
        ax[1,1].plot(vt2[:, i, 1], 'b')
        ax[2,1].plot(vt2[:, i, 2], 'b')

    error_p = torch.norm(pt - pt2)
    errot_v = torch.norm(vt - vt2)
    print('Mean Error in translation: {}, Max Error in translation {}'.format(error_p.mean(), error_p.max()))
    print('Mean Error in rotation: {}, Max Error in rotation {}'.format(errot_v.mean(), errot_v.max()))

    plt.show()


def test_train():
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_obs_dim = 2
    horizon = 12


    policy = load_model(horizon=horizon, obs_dim=n_obs_dim, device=device)

    B = 10
    ## Generate Random Action Trajectory ##
    act_r, act_p = policy.generate_random_initial_pose(B, horizon)

    ## Generate Observation ##
    obs_r, obs_p = policy.generate_random_initial_pose(B, 3)
    obs_f = torch.randn(B, 3, n_obs_dim)

    data ={
        'obs':{
            'obs_r': obs_r.to(device),
            'obs_p': obs_p.to(device),
            'obs_f': obs_f.to(device)
        },
        'action':{
            'act_r': act_r.to(device),
            'act_p':act_p.to(device)
        }
    }

    for t in range(100):
        time0 = time.time()
        loss = policy.compute_loss(data)
        dt = time.time() - time0
        print('step time:{}, 4 steps freq:{}'.format(dt, 1./(4*dt)))




if __name__ == "__main__":

    test_flow_and_vector_field()

    test_train()