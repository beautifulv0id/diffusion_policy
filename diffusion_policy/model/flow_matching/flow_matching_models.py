import torch
import torch.nn as nn
from diffusion_policy.model.common.so3_util import log_map,exp_map,random_so3

class EuclideanFlow(nn.Module):

    def __init__(self):
        super(EuclideanFlow, self).__init__()

    def generate_random_noise(self, batch=1, trj_steps=1, dim=9, device='cpu'):
        noise = torch.randn(batch, trj_steps, dim, device=device)
        return noise
    
    def flow_at_t(self, x0, x1, t):
        xt = x0 * (1. - t[:, None,None]) + x1 * t[:, None,None]
        return xt
    
    def vector_field_at_t(self, x1, xt, t):
        d_x = (x1 - xt) / torch.clip((1 - t[:, None, None]), 0.001, 1.)
        return d_x
    
    def step(self, _x, d_x, dt, time=None):
        x = _x + d_x * dt
        return x

class RectifiedLinearFlow(nn.Module):

    def __init__(self, world_translation=False):
        super(RectifiedLinearFlow, self).__init__()
        self.world_translation = world_translation

    def generate_random_initial_pose(self, batch=1, trj_steps=1, position_scale=1.):
        '''
        This function creates SE(3) homogeneous matrices
        :param batch_size: N, trj_steps T
        :return: a rotation NxTx3x3 and a translation NxTx3
        '''

        r = random_so3(batch*trj_steps).reshape(batch, trj_steps, 3, 3)
        t = torch.randn(batch, trj_steps, 3)*position_scale
        return r, t

    def flow_at_t(self, r0, p0, r1, p1, t):
        """
        Rectified Linear Attractor Flow
        Function which compute the sample xt along the geodesic from x0 to x1.
        """
        ## Translation ##
        pt = p0 * (1. - t[:, None, None]) + p1 * t[:, None, None]
        ## Rotation ##
        r0_inv = torch.transpose(r0, -1, -2)
        r10 = r0_inv@r1
        v = log_map(r10)
        vt = t[:,None,None]*v
        vt_rot = exp_map(vt)
        rt = r0@vt_rot

        return rt, pt

    def vector_field_at_t(self, r1, p1, rt, pt, t):
        """
        SE(3)-Rectified Linear Flow Vector Field
        """
        ## Translation ##
        d_pt = (p1 - pt) / torch.clip((1 - t[:, None, None]), 0.001, 1.)
        if self.world_translation is False:
            d_pt = torch.einsum('...mn,...n->...m', torch.transpose(rt,-1,-2), d_pt)
        ## Rotation ##
        r1t = torch.transpose(rt,-1,-2)@r1
        v = log_map(r1t)
        d_rt = v/torch.clip((1 - t[:, None, None]), 0.001, 1.)

        return d_rt, d_pt

    def step(self, _rt, _pt, d_rt, d_pt, dt, time=None, output_target=False):
        ## Translation ##
        if self.world_translation is False:
            d_pt = torch.einsum('...mn,...n->...m', _rt, d_pt)
        p1_pred = _pt + d_pt*(1-time)
        _pt = _pt + d_pt*dt
        ## Rotation ##
        r1_pred = _rt@exp_map(d_rt*(1-time))
        _rt = _rt@exp_map(d_rt*dt)
        if output_target:
            return r1_pred, p1_pred
        return _rt, _pt


class LinearAttractorFlow(nn.Module):
    '''
    Linear Attractor Flow applies a linear attractor for the first t<t_switch steps and then jumps to a Rectified Linear Flow.
    As a consideration, for maximum performance, be sure to align properly the number of steps with the t_switch value.
    For example, if the inference steps are 5, the time is (0., 0.25, 0.5, 0.75, 1.).
    Then, choose a t_switch in these values to adapt the switch perfectly. For example, Steps=5, t_switch=0.75
    '''
    def __init__(self, t_switch=0.5, output_target=False):
        super(LinearAttractorFlow, self).__init__()

        self.t_switch = t_switch
        self.control_gain = 1./(1.- t_switch)
        self.output_target = output_target

    def generate_random_initial_pose(self, batch=1, trj_steps=1, position_scale=1.):
        '''
        This function creates SE(3) homogeneous matrices
        :param batch_size: N, trj_steps T
        :return: a rotation NxTx3x3 and a translation NxTx3
        '''

        r = random_so3(batch * trj_steps).reshape(batch, trj_steps, 3, 3)
        t = torch.randn(batch, trj_steps, 3)*position_scale
        return r, t

    def flow_at_t(self, r0, p0, r1, p1, t):
        mask_t = t < self.t_switch

        ############## TRANSLATION ###############
        ##  t < self.t_switch ##
        pt_1 = p1 + (p0 - p1) * torch.exp(-self.control_gain * t[..., None, None])
        ## t > self.t_switch ##
        _pt = p1 + (p0 - p1) * torch.exp(-self.control_gain * self.t_switch * torch.ones_like(t[..., None, None]))
        t_after = t - self.t_switch
        pt2 = _pt * (1. - t[..., None, None]) / (1 - self.t_switch) + p1 * (t[..., None, None] - self.t_switch) / (
                1 - self.t_switch)
        pt = pt_1 * mask_t[..., None, None] + pt2 * (~mask_t)[..., None, None]

        ############## ROTATION ###############
        dt = t[..., None, None]
        r0_inv = torch.transpose(r0, -1, -2)
        r10 = r0_inv@r1
        v = log_map(r10)
        vt = dt*v
        vt_rot = exp_map(vt)
        rt = r0@vt_rot
        ########################################

        return rt, pt

    def vector_field_at_t(self, r1, p1, rt, pt, t):

        ## Translation ##
        d_pt = (p1 - pt) / torch.clip((1 - t[:, None, None]), 0.0, 1.)
        d_pt = torch.einsum('...mn,...n->...m', torch.transpose(rt,-1,-2), d_pt)

        ## Rotation ##
        r1t = torch.transpose(rt,-1,-2)@r1
        v = log_map(r1t)
        d_rt = v/torch.clip((1 - t[:, None, None]), 0.001, 1.)

        return d_rt, d_pt

    def step(self, _rt, _pt, d_rt, d_pt, dt, time=1, output_target=False):
        ## Translation ##
        d_pt = torch.einsum('...mn,...n->...m', _rt, d_pt)
        p1_pred = _pt + d_pt*(1-time)
        if time < self.t_switch:
            _pt = p1_pred + (_pt - p1_pred) * torch.exp(-self.control_gain * dt * torch.ones_like(_pt))
        else:
            _pt = _pt + d_pt * dt

        ## Rotation ##
        r1_pred = _rt@exp_map(d_rt*(1 - time))
        _rt = _rt @ exp_map(d_rt * dt)
        if output_target:
            return r1_pred, p1_pred
        return _rt, _pt


class SE3LinearAttractorFlow(nn.Module):
    '''
    Linear Attractor Flow applies a linear attractor for the first t<t_switch steps and then jumps to a Rectified Linear Flow.
    As a consideration, for maximum performance, be sure to align properly the number of steps with the t_switch value.
    For example, if the inference steps are 5, the time is (0., 0.25, 0.5, 0.75, 1.).
    Then, choose a t_switch in these values to adapt the switch perfectly. For example, Steps=5, t_switch=0.75
    '''
    def __init__(self, t_switch=0.5):
        super(SE3LinearAttractorFlow, self).__init__()

        self.t_switch = t_switch
        self.control_gain = 1./(1.- t_switch)

    def generate_random_initial_pose(self, batch=1, trj_steps=1, position_scale=1.):
        '''
        This function creates SE(3) homogeneous matrices
        :param batch_size: N, trj_steps T
        :return: a rotation NxTx3x3 and a translation NxTx3
        '''

        r = random_so3(batch * trj_steps).reshape(batch, trj_steps, 3, 3)
        t = torch.randn(batch, trj_steps, 3)*position_scale
        return r, t

    def flow_at_t(self, r0, p0, r1, p1, t):
        mask_t = t < self.t_switch

        ############## TRANSLATION ###############
        ##  t < self.t_switch ##
        pt_1 = p1 + (p0 - p1) * torch.exp(-self.control_gain * t[..., None, None])
        ## t > self.t_switch ##
        _pt = p1 + (p0 - p1) * torch.exp(-self.control_gain * self.t_switch * torch.ones_like(t[..., None, None]))
        t_after = t - self.t_switch
        pt2 = _pt * (1. - t[..., None, None]) / (1 - self.t_switch) + p1 * (t[..., None, None] - self.t_switch) / (
                1 - self.t_switch)

        pt = pt_1 * mask_t[..., None, None] + pt2 * (~mask_t)[..., None, None]

        ############## ROTATION ###############
        dt = t[..., None, None]
        r0_inv = torch.transpose(r0, -1, -2)
        r10 = r0_inv@r1
        v = log_map(r10)
        ##  t < self.t_switch ##
        _dt = 1 - torch.exp(-self.control_gain * t[..., None, None])
        _vt = _dt*v
        _vt_rot = exp_map(_vt)
        rt1 = r0@_vt_rot
        ##  t < self.t_switch ##
        _dt = 1 - torch.exp(-self.control_gain * self.t_switch * torch.ones_like(t[..., None, None]))
        _vt = _dt*v
        _vt_rot = exp_map(_vt)
        _rt_switch = r0@_vt_rot

        _t = t - self.t_switch
        gain = 1./(1.-self.t_switch)
        r0 = _rt_switch
        r0_inv = torch.transpose(r0, -1, -2)
        r10 = r0_inv@r1
        v = log_map(r10)
        _dt = _t[..., None, None]
        rt2 = r0@exp_map(v*_t[..., None, None]*gain)

        rt = rt1 * mask_t[..., None, None, None] + rt2 * (~mask_t)[..., None, None, None]
        ########################################

        return rt, pt

    def vector_field_at_t(self, r1, p1, rt, pt, t):

        mask_t = t < self.t_switch

        ## Translation ##
        ## We train both parts with the same velocity, to guarantee the size of the vectors is equal along the whole steps. We correct the velocity in step function .
        d_pt = (p1 - pt) / torch.clip((1 - t[:, None, None]), 0.0, 1.)
        d_pt = torch.einsum('...mn,...n->...m', torch.transpose(rt,-1,-2), d_pt)

        ## Rotation ##
        r1t = torch.transpose(rt,-1,-2)@r1
        v = log_map(r1t)
        ## We train both parts with the same velocity, to guarantee the size of the vectors is equal along the whole steps. We correct the velocity in step function .
        d_rt = v/torch.clip((1 - t[:, None, None]), 0.001, 1.)

        return d_rt, d_pt

    def step(self, _rt, _pt, d_rt, d_pt, dt, time=1, output_target=False):
        ## Translation ##
        d_pt = torch.einsum('...mn,...n->...m', _rt, d_pt)
        p1_pred = _pt + d_pt*(1-time)
        if time < self.t_switch:
            _pt = p1_pred + (_pt - p1_pred) * torch.exp(-self.control_gain * dt * torch.ones_like(_pt))
        else:
            _pt = _pt + d_pt * dt

        ## Rotation ##
        r1_pred = _rt@exp_map(d_rt*(1 - time))
        if time < self.t_switch:
            r0 = _rt
            r0_inv = torch.transpose(r0, -1, -2)
            r10 = r0_inv @ r1_pred
            v = log_map(r10)
            _dt = 1 - torch.exp(-self.control_gain * dt* torch.ones_like(_pt))
            _vt = _dt * v
            _vt_rot = exp_map(_vt)
            _rt = r0 @ _vt_rot
        else:
            _rt = _rt @ exp_map(d_rt * dt)

        if output_target:
            return r1_pred, p1_pred
        return _rt, _pt


############################################################################################################

def test(model, num_inference_steps=4):
    n_action_steps = 12

    num_inference_steps = num_inference_steps
    time_steps = num_inference_steps + 1
    ## FLOW EVALUATION ##
    r1 = torch.eye(3)[None, None, ...].repeat(time_steps, n_action_steps, 1, 1)
    p1 = torch.zeros(time_steps, n_action_steps, 3)

    r0, p0 = model.generate_random_initial_pose(batch=1, trj_steps=n_action_steps)
    r0, p0 = r0[:, ...].repeat(time_steps, 1, 1, 1), p0[:, ...].repeat(time_steps, 1, 1)

    time = torch.linspace(0, 1, time_steps)
    rt, pt = model.flow_at_t(r0, p0, r1, p1, time)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=3, ncols=2)

    vt = log_map(rt)
    for i in range(n_action_steps):
        ax[0,0].plot(pt[:, i, 0], 'r')
        ax[1,0].plot(pt[:, i, 1], 'r')
        ax[2,0].plot(pt[:, i, 2], 'r')

        ax[0,1].plot(vt[:, i, 0], 'r')
        ax[1,1].plot(vt[:, i, 1], 'r')
        ax[2,1].plot(vt[:, i, 2], 'r')


    ## VECTOR FIELD EVALUATION ##
    _rt, _pt = r0[0, ...], p0[0, ...]
    _r1, _p1 = r1[0, ...], p1[0, ...]
    trj_r, trj_p = _rt[None, ...], _pt[None, ...]

    dt = 1 / num_inference_steps
    for t in range(0, num_inference_steps):
        print(t)
        d_rt, d_pt = model.vector_field_at_t(_r1[None, ...], _p1[None, ...], _rt[None, ...], _pt[None, ...],
                                              dt * t * torch.ones_like(_p1[:1, 0]))
        _rt, _pt = model.step(_rt, _pt, d_rt[0, ...], d_pt[0, ...], dt, time=t*dt)

        ## Save in Trajectory ##
        trj_r, trj_p = torch.cat((trj_r, _rt[None, ...]), dim=0), torch.cat((trj_p, _pt[None, ...]), dim=0)

    ## Visualize ##
    vt2 = log_map(trj_r)
    pt2 = trj_p
    for i in range(n_action_steps):
        ax[0,0].plot(pt2[:, i, 0], 'b')
        ax[1,0].plot(pt2[:, i, 1], 'b')
        ax[2,0].plot(pt2[:, i, 2], 'b')

        ax[0,1].plot(vt2[:, i, 0], 'b')
        ax[1,1].plot(vt2[:, i, 1], 'b')
        ax[2,1].plot(vt2[:, i, 2], 'b')


    titles = ['Trans X', 'Rotat X', 'Trans Y', 'Rotat Y', 'Trans Z', 'Rotat Z']
    for i, axi in enumerate(ax.flat):
        axi.set_title(titles[i])

    error_p = torch.norm(pt - pt2)
    errot_v = torch.norm(vt - vt2)
    print('Mean Error in translation: {}, Max Error in translation {}'.format(error_p.mean(), error_p.max()))
    print('Mean Error in rotation: {}, Max Error in rotation {}'.format(errot_v.mean(), errot_v.max()))
    plt.show()

def test_euclidean_flow():
    from tqdm import tqdm
    from diffusion_policy.model.common.position_encodings import SinusoidalPosEmb
    num_inference_steps = 100
    device = 'cuda'
    flow_model = EuclideanFlow().to(device)

    class LinearPredictor(nn.Module):
        def __init__(self, time_embed_dim=64):
            super(LinearPredictor, self).__init__()
            self.predictor = nn.Sequential(nn.Linear(9+time_embed_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 9))
            self.time_embedder = SinusoidalPosEmb(time_embed_dim)
        def forward(self, x, t):
            t = self.time_embedder(t)[:,None,:]
            x = torch.cat((x, t), dim=-1)
            return self.predictor(x)
        
        
    predictor = LinearPredictor()
    predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    B = 1
    n_action_steps = 1
    rot = random_so3(B * n_action_steps).reshape(B, n_action_steps, 3, 3)
    rot_6D = rot[...,:2].reshape(B, n_action_steps, 6)
    pos = torch.randn(B, n_action_steps, 3)
    x1 = torch.cat((pos, rot_6D), dim=-1).reshape(B, n_action_steps, 9)

    x1_stacked = x1.repeat(2048, 1, 1)
    x1_stacked = x1_stacked.to(device)
    print(x1_stacked.shape)

    with tqdm(range(10000)) as epoch:
        for i in epoch:
            optimizer.zero_grad()
            x0 = flow_model.generate_random_noise(batch=2048, trj_steps=n_action_steps, dim=9, device=device)
            time = torch.rand(2048, device=device)
            xt = flow_model.flow_at_t(x0, x1_stacked, time)
            d_x = flow_model.vector_field_at_t(x1_stacked, xt, time)
            d_x_pred = predictor(xt, time)
            loss = torch.functional.F.mse_loss(d_x, d_x_pred)
            loss.backward()
            optimizer.step()
            epoch.set_postfix(loss=loss.cpu().item())

    
    with torch.no_grad():
        dt = 1 / num_inference_steps
        x0 = flow_model.generate_random_noise(batch=B, trj_steps=n_action_steps, dim=9).to(device)
        xt = x0
        for s in range(0, num_inference_steps):
            time = s * dt * torch.ones(B).to(device)
            d_x = predictor(xt, time)
            xt = flow_model.step(xt, d_x, dt, time)

    print(torch.norm(xt.cpu() - x1))
    print(xt.cpu())
    print(x1)
    print('Done')
    

if __name__ == '__main__':
    test_euclidean_flow()

    # model = RectifiedLinearFlow()
    # num_inference_steps = 4
    # test(model, num_inference_steps=num_inference_steps)

    # model = LinearAttractorFlow(t_switch=.75)
    # num_inference_steps = 4
    # test(model, num_inference_steps=num_inference_steps)

    # model = SE3LinearAttractorFlow(t_switch=.75)
    # num_inference_steps = 4
    # test(model, num_inference_steps=num_inference_steps)