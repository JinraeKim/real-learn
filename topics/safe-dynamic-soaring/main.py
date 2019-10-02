import numpy as np
import matplotlib.pyplot as plt

import torch

from envs import SoaringEnv
from agents import Agent
from utils import Differentiator


def main():
    np.random.seed(1)

    time_step = 0.005
    time_series = np.arange(0, 2, time_step)

    env = SoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    agent = Agent(env, time_step)
    agent.diff = Differentiator()

    obs = env.reset()
    obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

    for i, t in enumerate(time_series):
        controls = [0, 1, 0]

        next_obs, reward, done, info = env.step(controls)

        agent.diff.append(t, obs)

        if i % agent.sampling_interval == 1:
            x, y = agent.diff.get()
            agent.dataset.append(x, y)

        if done:
            break

        obs = next_obs
        obs_series = np.vstack((obs_series, obs))


def train_sample():
    np.random.seed(1)
    torch.manual_seed(1)

    time_step = 0.005

    env = SoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    agent = Agent(env, lr=1e-4, time_step=time_step)

    low = np.hstack((-10, 3, np.deg2rad([-40, -50])))
    high = np.hstack((-1, 15, np.deg2rad([40, 50])))
    dataset = np.random.uniform(low=low, high=high, size=(100000, 4))
    dataset = torch.tensor(dataset).float()

    agent.dataset = dataset

    agent.train_safe_value(verbose=1)

    # Saving model
    torch.save(agent.safe_value.state_dict(), 'model.pth')


if __name__ == '__main__':
    if False:
        train_sample()

    time_step = 0.005
    time_series = np.arange(0, 2, time_step)

    env = SoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    agent = Agent(env, lr=1e-4, time_step=time_step)
    agent.safe_value.load_state_dict(torch.load('model.pth'))

    # Evaluation
    low = np.hstack((-10, 3, np.deg2rad([-40, -50])))
    high = np.hstack((-0.03, 15, np.deg2rad([40, 50])))

    agent.safe_value.eval()
    z, V = np.meshgrid(*np.linspace(low[:2], high[:2], 20).T)
    gamma = np.ones_like(V) * np.deg2rad(0)
    psi = np.ones_like(V) * np.deg2rad(0)
    value = np.vectorize(agent.safe_value.from_numpy)
    constraint = np.vectorize(agent.const_func)
    s = value(z, V, gamma, psi)

    # fig, ax = plt.subplots(1, 1)
    # ax.contour(z, V, s)
    # plt.show()
