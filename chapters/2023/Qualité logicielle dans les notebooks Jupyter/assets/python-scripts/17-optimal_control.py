#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from torch.optim import SGD
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl
import numpy as np

# In[ ]:


plt.style.use(['dark_background', 'bmh'])
plt.rc('axes', facecolor='k')
plt.rc('figure', facecolor='k', figsize=(10, 6), dpi=100)  # (17, 10)
plt.rc('savefig', bbox='tight')
plt.rc('axes', labelsize=36)
plt.rc('legend', fontsize=24)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
plt.rc('lines', markersize=10)

# The state transition equation is the following:
# 
# $$\def \vx {\boldsymbol{\color{Plum}{x}}}
# \def \vu {\boldsymbol{\color{orange}{u}}}
# \dot{\vx} = f(\vx, \vu) \quad
# \left\{
# \begin{array}{l}
# \dot{x} = s \cos \theta \\
# \dot{y} = s \sin \theta \\
# \dot{\theta} = \frac{s}{L} \tan \phi \\
# \dot{s} = a
# \end{array}
# \right. \quad
# \vx = (x\;y\;\theta\;s) \quad
# \vu = (\phi\;a)
# $$

# In[ ]:


def f(x, u, t=None):
    """
    Kinematic model for tricycle
    ẋ(t) = f[x(t), u(t), t]
    x: states (x, y, θ, s)
    u: control
    t: time
    f: kinematic model
    ẋ = dx/dt
    x' = x + f(x, u, t) * dt
    """
    L = 1  # m
    x, y, θ, s = x
    
    ϕ, a = u
    f = torch.zeros(4)
    f[0] = s * torch.cos(θ)
    f[1] = s * torch.sin(θ)
    f[2] = s / L * torch.tan(ϕ)
    f[3] = a
    return f

# In[ ]:


def draw_car(ax, x, y, θ, width=0.4, length=1.0):
    rect = patches.Rectangle(
        (x, y - width / 2), 
        length,
        width,
        transform=mpl.transforms.Affine2D().rotate_around(*(x, y), θ) + ax.transData,
        alpha=0.8,
        fill=False,
        ec='grey',
    )
    ax.add_patch(rect)
    
def plot_τ(ax, τ, car=False, ax_lims=None):
    """
    Plot trajectory of vehicles
    ax_lims is a tuple of two tuples ((x_lim_left, x_lim_right), (y_lim_bottom, y_lim_top))
    """
    if ax_lims is None:
        ax_lims = ((-1, 7), (-2, 2))
    ax.plot(τ[:,0], τ[:,1], 'o-')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.autoscale(False)
    ax.set_xlabel(r'$x \; [\mathrm{m}]$')
    ax.set_ylabel(r'$y \; [\mathrm{m}]$')
    
    ax.set_xlim(*ax_lims[0])
    ax.set_ylim(*ax_lims[1])
    ax.set_xticks(torch.arange(ax_lims[0][0], ax_lims[0][1] + 1, 1))
    ax.set_yticks(torch.arange(ax_lims[1][0], ax_lims[1][1] + 1, 1))
        
    plt.title('Trajectory')
    if car:
        for x, y, θ in τ[:, :3]:
            draw_car(plt.gca(), x, y, θ)

# In[ ]:


# Manual driving
x = torch.tensor((0, 0, 0, 1),dtype=torch.float32)
# Optimal action from back propagation
u = torch.tensor([
    [0.1280, 0.0182],
    [0.0957, 0.0131],
    [0.0637, 0.0085],
    [0.0318, 0.0043],
    [0.0000, 0.0000]
])
# Brake
u = torch.ones(10, 2) * -0.1
u[:, 0] = 0
# S
u = torch.zeros(10, 2)
u[:5, 0] = 0.2
u[5:, 0] = -0.2
# Straight
# u = torch.zeros(10, 2)

dt = 1  # s
trajectory = [x.clone()]
for t in range(10):
    x += f(x, u[t]) * dt
    print(x)
    trajectory.append(x.clone())
τ = torch.stack(trajectory)

# plt.plot(0,0,'gx', markersize=20, markeredgewidth=5)
# plt.plot(5,1,'rx', markersize=20, markeredgewidth=5)
plot_τ(plt.gca(), τ, car=True)

plt.axis((-1, 10, -1, 5))
name = 'S'

# plt.axis((-1, 12, -3, 3))
# name = 'straight'

# plt.axis((-1, 7, -2, 2))
# name = 'brake'

# plt.savefig(f'{name}.pdf')

plt.figure(figsize=(6, 2))
plt.title('Control signal')
plt.stem(np.arange(10)+0.9, u[:,0], 'C1', markerfmt='C1o', use_line_collection=True, basefmt='none')
plt.stem(np.arange(10)+1.1, u[:,1], 'C2', markerfmt='C2o', use_line_collection=True, basefmt='none')
plt.ylim((-0.5, 0.5))
plt.xticks(np.arange(12))
plt.xlabel('discrete time index', fontsize=12)
# plt.savefig(f'{name}-ctrl.pdf')

# In[ ]:


# Costs definition
# x: states (x, y, θ, s)
def vanilla_cost(state, target):
    x_x, x_y = target
    return (state[-1][0] - x_x).pow(2) + (state[-1][1] - x_y).pow(2)

def cost_with_target_s(state, target):
    x_x, x_y = target
    return (state[-1][0] - x_x).pow(2) + (state[-1][1] - x_y).pow(2) + (state[-1][-1]).pow(2)

def cost_sum_distances(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2)).pow(0.5)
    return dists.mean()

def cost_sum_square_distances(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2))
    return dists.mean()

def cost_logsumexp(state, target):
    x_x, x_y = target
    dists = ((state[:, 0] - x_x).pow(2) + (state[:, 1] - x_y).pow(2))#.pow(0.5)
    return -1 * torch.logsumexp(-1 * dists, dim=0)

# In[ ]:


# Path planning
def path_planning_with_cost(x_x, x_y, s, T, epochs, stepsize, cost_f, ax=None, ax_lims=None, debug=False):
    """
    Path planning for tricycle
    x_x: x component of postion vector
    x_y: y component of postion vector
    s: initial speed
    T: time steps
    epochs: number of epochs for back propagation
    stepsize: stepsize for back propagation
    cost_f: cost funciton that takes the trajectory and the tuple (x, y) - target.
    ax: axis to plot the trajectory
    """
    ax = ax or plt.gca()
    plt.plot(0, 0, 'gx', markersize=20, markeredgewidth=5)
    plt.plot(x_x, x_y, 'rx', markersize=20, markeredgewidth=5)
    u = nn.Parameter(torch.zeros(T, 2))
    optimizer = SGD((u,), lr=stepsize)
    dt = 1  # s
    costs = []
    for epoch in range(epochs):
        x = [torch.tensor((0, 0, 0, s),dtype=torch.float32)]
        for t in range(1, T+1):
            x.append(x[-1] + f(x[-1], u[t-1]) * dt)
        x_t = torch.stack(x)
        τ = torch.stack(x).detach()
        cost = cost_f(x_t, (x_x, x_y))
        costs.append(cost.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if debug: 
            print(u.data)
        # Only plot the first and last trajectories
        if epoch == 0: 
            plot_τ(ax, τ, ax_lims=ax_lims)
        if epoch == epochs-1:
            plot_τ(ax, τ, car=True, ax_lims=ax_lims)

# In[ ]:


path_planning_with_cost(x_x=5, x_y=1, s=1, T=5, epochs=5, stepsize=0.01, cost_f=vanilla_cost, debug=False)

# In[ ]:


plt.figure(dpi=100, figsize=(10, 55))
for i in range(5, 16):
    ax = plt.subplot(11, 1, i - 5 + 1)
    path_planning_with_cost(x_x=5, x_y=1, s=1, T=i, epochs=50, stepsize=0.001, ax=ax, cost_f=vanilla_cost, debug=False)
    plt.title(f'T={i}')
plt.tight_layout()
plt.suptitle('Using just final position for the cost', y=1.01)
# plt.savefig('final-position.pdf')

# In[ ]:


plt.figure(dpi=100, figsize=(10, 55))
plt.suptitle('Using final position and speed for the cost', y=1.01)
for i in range(5, 16):
    ax = plt.subplot(11, 1, i - 5 + 1)
    path_planning_with_cost(x_x=5, x_y=1, s=1, T=i, epochs=50, stepsize=0.001, cost_f=cost_with_target_s, ax=ax, debug=False)
    plt.title(f"T={i}")
plt.tight_layout()
# plt.savefig('final-position-and-speed.pdf')

# In[ ]:


plt.figure(dpi=100, figsize=(10, 55))
plt.suptitle('Using sum of distances for the cost', y=1.01)
for i in range(5, 16):
    ax = plt.subplot(11, 1, i - 5 + 1)
    costs = path_planning_with_cost(x_x=5, x_y=1, s=1, T=i, epochs=40, stepsize=0.0025, ax=ax, cost_f=cost_sum_square_distances, debug=False)
    plt.title(f"T={i}")
    plt.gca().set_aspect("equal")
plt.tight_layout()
# plt.savefig('average-distance.pdf')

# In[ ]:


plt.figure(dpi=100, figsize=(10, 55))
plt.suptitle('Using softmin of distances for the cost (focusing on the points closest to target)', y=1.01)
for i in range(5, 16):
    ax = plt.subplot(11, 1, i - 5 + 1)
    path_planning_with_cost(x_x=5, x_y=1, s=1, T=i, epochs=100, stepsize=0.005, cost_f=cost_logsumexp, ax=ax, debug=False)
    plt.title(f"T={i}")
plt.tight_layout()
plt.savefig('softmin.pdf')
