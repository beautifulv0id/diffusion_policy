import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm.auto import tqdm

## Geometry Related Libraries ##
os.environ['GEOMSTATS_BACKEND']='pytorch'
from geomstats._backend import _backend_config as _config
_config.DEFAULT_DTYPE = torch.FloatTensor
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from scipy.spatial.transform import Rotation

## Torch Related
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

## Dataset ##
from diffusion_policy.dataset.special_euclidean_group import ConditionedSpecialEuclideanGroup

## Utils
from diffusion_policy.common.plotting import plot_se3
from diffusion_policy.common.pytorch_util import dict_apply

## Model
from geo_rel_policies.policy.feat_pcloud_naive_policy import NaivePolicy


NOVEL_ROTATIONS_IN_EVAL = True
SINGLE_POINT = True
VELOCITY_SCALE = 2.

class SE3FlowMatching(nn.Module):
    def __init__(self, n_context, velocity_scale=1.):
        super(SE3FlowMatching, self).__init__()

        self.model = NaivePolicy(n_context=n_context, dim=16, output_size=6)

        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

        ## Hyperparameters
        self.velocity_scale = velocity_scale

    def forward(self, x, t):
        out = self.model(x, t)
        return out

    def set_context(self, context):
        self.model.set_context(context)

    def sample(self, batch_size=64, T=100, get_traj=False):

        with torch.no_grad():
            steps = T
            t = torch.linspace(0, 1., steps=steps)
            dt = 1/steps

            # Euler method
            # sample H_0 first
            x0 = self.get_random_pose(batch_size=batch_size)

            if get_traj:
                trj = x0[:,None,...].repeat(1,steps,1,1)
            xt = x0
            for s in range(0, steps):
                ut = self.velocity_scale*self.forward(xt, t[s]*torch.ones_like(xt[:,0,0]))
                utdt = ut*dt

                ## rotation update ##
                rot_xt = xt[:, :3, :3]
                rot_utdt = utdt[:, :3]
                rot_xt_v = self.vec_manifold.rotation_vector_from_matrix(rot_xt)
                rot_xt_v2 = self.vec_manifold.compose(rot_xt_v, rot_utdt)
                rot_xt_2 = self.vec_manifold.matrix_from_rotation_vector(rot_xt_v2)

                ## translation update ##
                trans_xt = xt[:, :3, -1]
                trans_utdt = utdt[:, 3:]
                #Place the translation back in world frame
                #trans_utdt = torch.einsum('bij,bj->bi', rot_xt.transpose(-1,-2), trans_utdt)
                trans_xt2 = trans_xt + trans_utdt

                xt[:, :3, :3] = rot_xt_2
                xt[:, :3, -1] = trans_xt2

                if get_traj:
                    trj[:,s,...] = xt
        if get_traj:
            return xt, trj
        else:
            return xt

    def get_random_pose(self, batch_size):
        R = torch.tensor(Rotation.random(batch_size).as_matrix()).to(device)
        t = torch.randn(batch_size, 3).to(device)
        H = torch.eye(4)[None,...].repeat(batch_size,1,1).to(device)
        H[:, :3, :3] = R
        H[:, :3, -1] = t
        return H

    def compute_loss(self, x1, context):

        self.set_context(context)

        x0 = self.get_random_pose(x1.size(0))
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)

        ## Sample X at time t through the Geodesic from x0 -> x1
        def sample_xt(x0, x1, t):
            """
            Function which compute the sample xt along the geodesic from x0 to x1 on SE(3).
            """
            ## Point to translation and rotation ##
            t0, R0 = x0[:, :3, -1], x0[:, :3, :3]
            t1, R1 = x1[:, :3, -1], x1[:, :3, :3]

            ## Get rot_t ##
            rot_x0 = self.vec_manifold.rotation_vector_from_matrix(R0)
            rot_x1 = self.vec_manifold.rotation_vector_from_matrix(R1)
            log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
            rot_xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
            rot_xt = self.vec_manifold.matrix_from_rotation_vector(rot_xt)

            ## Get trans_t ##
            trans_xt = t0*(1. - t[:,None]) + t1*t[:,None]

            xt = torch.eye(4)[None,...].repeat(rot_xt.shape[0], 1, 1).to(device)
            xt[:, :3, :3] = rot_xt
            xt[:, :3, -1] = trans_xt
            return xt

        xt = sample_xt(x0.double(), x1.double(), t)

        ## Compute velocity target at xt at time t through the geodesic x0 -> x1
        def compute_conditional_vel(x0, x1, xt, t):

            def invert_se3(matrix):
                """
                Invert a homogeneous transformation matrix.

                :param matrix: A 4x4 numpy array representing a homogeneous transformation matrix.
                :return: The inverted transformation matrix.
                """

                # Extract rotation (R) and translation (t) from the matrix
                R = matrix[..., :3, :3]
                t = matrix[..., :3, 3]

                # Invert the rotation (R^T) and translation (-R^T * t)
                R_inv = torch.transpose(R, -1, -2)
                t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)

                # Construct the inverted matrix
                inverted_matrix = torch.clone(matrix)
                inverted_matrix[..., :3, :3] = R_inv
                inverted_matrix[..., :3, 3] = t_inv

                return inverted_matrix


            xt_inv = invert_se3(xt)

            xt1 = torch.einsum('...ij,...jk->...ik', xt_inv, x1)
            ## Point to translation and rotation ##
            trans_xt1, rot_xt1 = xt1[:, :3, -1], xt1[:, :3, :3]

            trans_xt, rot_xt = xt[:, :3, -1], xt[:, :3, :3]
            trans_x1, rot_x1 = x1[:, :3, -1], x1[:, :3, :3]

            ## Compute Velocity in rot ##
            delta_r = torch.transpose(rot_xt, -1, -2)@rot_x1
            rot_ut = self.vec_manifold.rotation_vector_from_matrix(delta_r) / torch.clip((1 - t[:, None]), 0.01, 1.)

            ## Compute Velocity in trans ##
            #trans_ut = -trans_xt1/ torch.clip((1 - t[:, None]), 0.01, 1.)
            trans_ut = (trans_x1 - trans_xt)/ torch.clip((1 - t[:, None]), 0.01, 1.)
            #trans_ut = 0.*trans_ut


            return torch.cat((rot_ut, trans_ut), dim=1).detach()

        ut = compute_conditional_vel(x0.double(), x1.double(), xt.double(), t)

        xt.requires_grad = True
        t.requires_grad = True
        vt = self.forward(xt, t)

        loss = torch.mean((ut-vt)**2)
        return loss

    def get_metrics(self, r0, r1):
        t0, r0 = r0[:, :3, -1], r0[:, :3, :3]
        t1, r1 = r1[:, :3, -1], r1[:, :3, :3]


        _r0 = self.vec_manifold.rotation_vector_from_matrix(r0)
        _r1 = self.vec_manifold.rotation_vector_from_matrix(r1)
        log_10 = self.vec_manifold.log_not_from_identity(_r0, _r1)
        distance = torch.norm(log_10, dim=-1).mean()
        print('SO3 distance',distance)
        print('R3 distance', (t0-t1).pow(2).sum(-1).pow(.5).mean())


def train_loop(model, optimizer, num_epochs=10000, display=True):
    losses = []
    w1ds = []
    w2ds = []
    global_step = 0

    for epoch in range(num_epochs):

        if display:
            progress_bar = tqdm(total=len(trainloader))
            progress_bar.set_description(f"Epoch {epoch}")

        if (epoch % 10) == 0:

            if SINGLE_POINT:
                batch_size = 200
                context_points = testset.points

                context_points = torch.Tensor(context_points)[None,...].repeat(batch_size, 1, 1).to(device)
                model.set_context(context_points)
                x = model.sample(batch_size=batch_size, get_traj=False)

                ## Distance to single data point ##
                datapoint = trainset.data[0]
                datapoint = torch.Tensor(datapoint)[None,...].repeat(batch_size, 1, 1).to(device)

                model.get_metrics(x, datapoint)

                if display:
                    plot_se3(datapoint)
                    plot_se3(x)
                    plt.show()

            else:
                batch_size = 2000

                y = testset.data
                context_points = testset.points

                rot = testset.get_random_rotation()
                rotated_points = np.einsum('mn,bn->bm', rot, context_points)
                y = np.einsum('mn,bnk->bmk', rot, y)


                context_points = torch.Tensor(rotated_points)[None,...].repeat(batch_size, 1, 1).to(device)

                model.set_context(context_points)

                x = model.sample(batch_size=batch_size, get_traj=False)

                #rotated_x = x.detach().numpy()
                #x = np.einsum('mn,bnk->bmk', rot.transpose(-1,-2), rotated_x)

                if display:
                    plot_se3(x)
                    plot_se3(y)
                    plt.show()


        for _, batch in enumerate(trainloader):
            optimizer.zero_grad()

            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            x1 = batch['data']
            context = batch['points']

            loss = model.compute_loss(x1, context)
            losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            if display:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

    return model, np.array(losses), np.array(w1ds), np.array(w2ds)



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'


    ### DATASET ###
    trainset = ConditionedSpecialEuclideanGroup(split="train",
                                                 random_rotations=not(NOVEL_ROTATIONS_IN_EVAL), single_point=SINGLE_POINT)
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=0)
    # valset = ConditionedSpecialEuclideanGroup(split="valid")
    # valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=0)
    testset = ConditionedSpecialEuclideanGroup(split="test",
                                                random_rotations=not(NOVEL_ROTATIONS_IN_EVAL), single_point=SINGLE_POINT)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)


    ## LOAD MODEL ##
    n_context = trainset.context_shape[0]
    model = SE3FlowMatching(n_context=n_context, velocity_scale=VELOCITY_SCALE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_loop(model, optimizer)



