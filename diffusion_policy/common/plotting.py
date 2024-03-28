import numpy as np
import math
import matplotlib.pyplot as plt

################################ SO(3) ################################

def plot_scatter3D(xyz, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)):
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5)
    ax.axes.set_xlim3d(xlim[0], xlim[1])
    ax.axes.set_ylim3d(ylim[0], ylim[1])
    ax.axes.set_zlim3d(zlim[0], zlim[1])
    return fig


def rotationMatrixToEulerAngles(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def plot_so3(x):
    return plot_scatter3D(
            np.array([rotationMatrixToEulerAngles(x[i]) for i in range(len(x))]),
            (-math.pi, math.pi),
            (-math.pi / 2, math.pi / 2),
            (-math.pi, math.pi),
        )



################################ SE(3) ################################

def plot_rotated_axes(ax, r, name=None, offset=np.zeros(3), scale=1, color=None):

        if color is None:
            colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
        elif color =='actor':
            colors = ("#000000", "#AAAAAA", "#1199EE")  # Colorblind-safe RGB
        else:
            colors = ("#EE5555", "#004422", "#0088DD")  # Colorblind-safe RGB



        for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                          colors)):
            axlabel = axis.axis_name
            axis.set_label_text(axlabel)
            axis.label.set_color(c)
            axis.line.set_color(c)
            axis.set_tick_params(colors=c)

            line = np.zeros((2, 3))
            line[1, :] = r[:, i]*0.1
            line_plot = line + offset
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)


def plot_se3(H, ax=None, color=None):
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")

    if isinstance(H, (np.ndarray, np.generic))==False:
        H = H.numpy()
    R = H[:, :3, :3]
    X = H[:, :3, -1]
    for k in range(H.shape[0]):
        plot_rotated_axes(ax, R[k, ...], offset=X[k, ...], color=color)

    return ax

def plot_3d(X, ax=None, color='green'):
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")

    if isinstance(X, (np.ndarray, np.generic))==False:
        X = X.numpy()

    ax.scatter3D(X[:,0], X[:,1], X[:,2], color=color)
    return ax
