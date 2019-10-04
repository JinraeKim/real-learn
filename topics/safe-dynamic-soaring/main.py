import numpy as np
import matplotlib.pyplot as plt

import torch

from envs import SoaringEnv
from agents import Agent, SafeValue
from utils import Differentiator


def train(save_name):
    np.random.seed(1)
    torch.manual_seed(1)

    env = SoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=0.005
    )
    agent = Agent(env, lr=1e-4)

    low = np.hstack((-10, 3, np.deg2rad([-40, -50])))
    high = np.hstack((-1, 15, np.deg2rad([40, 50])))
    dataset = np.random.uniform(low=low, high=high, size=(1000000, 4))
    dataset = torch.tensor(dataset).float()

    agent.dataset = dataset

    agent.train_safe_value(verbose=1)

    # Saving model
    torch.save(agent.safe_value.state_dict(), save_name)


class PlotVar:
    def __init__(self, latex, bound, is_angle=False, desc=None):
        self.latex = latex
        self.bound = bound
        self.is_angle = is_angle
        self.desc = desc

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        self.axis = np.rad2deg(grid) if self.is_angle else grid


z = PlotVar(
    desc='negative altitude', latex=r'$z$ (m)', bound=[-10, -0.01],)
V = PlotVar(
    desc='air speed', latex=r'$V$ (m/s)', bound=[3, 15],)
gamma = PlotVar(
    desc='negative altitude', latex=r'$\gamma$ (deg)',
    bound=np.deg2rad([-45, 45]), is_angle=True,)
psi = PlotVar(
    desc='negative altitude', latex=r'$\psi$ (deg)',
    bound=np.deg2rad([-60, 60]), is_angle=True,)


def draw_safe_value(name, model_file, pvars, fvars, fvals):
    safe_value = SafeValue()
    safe_value.load_state_dict(torch.load(model_file))
    safe_value.eval()

    # Evaluation points (z, V, gamma, psi)
    def gen_grid(pvars, fvars, fvals, num=100):
        pvars[0].grid, pvars[1].grid = np.meshgrid(
            np.linspace(*pvars[0].bound, num),
            np.linspace(*pvars[1].bound, num)
        )

        for fvar, fval in zip(fvars, fvals):
            fvar.grid = np.ones_like(pvars[0].grid) * fval

    gen_grid(pvars=pvars, fvars=fvars, fvals=fvals)

    # Evaluate the safe value for each data point
    value = np.vectorize(safe_value.from_numpy)
    s = value(*map(lambda x: x.grid, [z, V, gamma, psi]))
    s = np.ma.array(s, mask=s < 0.)

    # Draw a plot
    fig, ax = plt.subplots(1, 1)
    ax.contour(
        pvars[0].axis, pvars[1].axis, s,
        levels=14, linewidths=0.5, colors='k')
    cntr = ax.contourf(
        pvars[0].axis, pvars[1].axis, s,
        levels=14, cmap='RdBu_r')
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel(pvars[0].latex)
    ax.set_ylabel(pvars[1].latex)
    fig.tight_layout()
    plt.show()
    fig.savefig(name)


if __name__ == '__main__':
    # train('model.pth')
    draw_safe_value(
        name='z-and-V.png', model_file='model.pth',
        pvars=(z, V), fvars=(gamma, psi), fvals=(0.25, 0.25))
    draw_safe_value(
        name='gamma-and-psi.png', model_file='model.pth',
        pvars=(gamma, psi), fvars=(z, V), fvals=(-2.4, 8))
