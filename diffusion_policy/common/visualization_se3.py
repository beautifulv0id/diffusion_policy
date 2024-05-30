import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import plotly.graph_objs as go
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.so3_util import log_map

def create_surface_plane():
    obj = go.Mesh3d(
        # 8 vertices of a cube
        x=[-1, -1, .2, .2, -1, -1, .2, .2],
        y=[-.5, .5, .5, -.5, -.5, .5, .5, -.5],
        z=[-0.03, -0.03, -0.03, -0.03, -0.025, -0.025, -0.025, -0.025],
        # i, j and k give the vertices of triangles
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='plane',
        color='rgb(190,230,255)',
    )
    return obj


def create_frame_trace(rotations, translations, width=6, opacity=1.0, color=['red', 'green', 'blue'], length=0.1):
    traces = []

    steps = rotations.shape[0]
    for t in range(rotations.shape[0]):
        progress = t/steps
        # if t%2 != 0 and t%3 != 0:
        #     continue
        # Extract the rotation and translation from the transformation matrix
        rotation = rotations[t,...]
        translation = translations[t,...]

        # Create a 3D line trace for each axis
        for j in range(3):
            axis = rotation[:, j]
            end = translation + length*axis
            trace = go.Scatter3d(
                x=[translation[0], end[0]],
                y=[translation[1], end[1]],
                z=[translation[2], end[2]],
                mode='lines',
                line=dict(color=color[j], width=width,
                ),opacity=opacity
            )
            traces.append(trace)

    return traces


def visualize_frames(rotations, translations, width=6, opacity=1.0, color=['red', 'green', 'blue'], length=0.1):
    # Create a figure
    fig = go.Figure()

    # Add the surface plane
    fig.add_trace(create_surface_plane())

    # Add the frame traces
    traces = create_frame_trace(rotations, translations, width=width, opacity=opacity, color=color, length=length)
    for trace in traces:
        fig.add_trace(trace)

    # Set the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-0.2, 3.]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )

    # Show the figure
    fig.show()


def visualize_poses_and_actions(state_rotations, state_translations, action_rotations, action_translations, width=6, opacity=1.0, color=['red', 'green', 'blue']):
    # Create a figure
    fig = go.Figure()

    # Add the surface plane
    fig.add_trace(create_surface_plane())


    # Add the state traces
    traces = create_frame_trace(state_rotations, state_translations, width=width*2, opacity=opacity, color=['crimson', 'darkolivegreen', 'teal'])
    for trace in traces:
        fig.add_trace(trace)

    # Add the frame traces
    traces = create_frame_trace(action_rotations, action_translations, width=width, opacity=opacity/2, color=color, length=0.03)
    for trace in traces:
        fig.add_trace(trace)

    # Set the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-0.2, 3.]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
    )

    # Show the figure
    fig.show()