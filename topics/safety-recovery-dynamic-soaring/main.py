import numpy as np
import matplotlib.pyplot as plt
from fym.utils.plotting import PltModule

from envs import DynamicSoaringEnv
from agents import Agent


def main():
    time_step = 0.005
    time_series = np.arange(0, 2, time_step)

    env = DynamicSoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=time_step
    )
    obs = env.reset()
    obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

    agent = Agent(env)

    for i, t in enumerate(time_series):
        controls = agent.get()

        next_obs, reward, done, info = env.step(controls)

        if done:
            break

        obs = next_obs
        obs_series = np.vstack((obs_series, obs))

    time_series = time_series[:obs_series.shape[0]]

    if False:
        data = {
        'traj': obs_series[:, 0:3],
        'xyz': obs_series[:, 0:3]
        }

        variables = {
            'traj': ('x', 'y', 'z'),
            'xyz': ('x', 'y', 'z'),
        }

        quantities = {
            'traj': ('distance', 'distance', 'distance'),
            'xyz': ('distance', 'distance', 'distance'),
        }

        labels = ('traj', 'xyz')

        figures = PltModule(time_series, data, variables, quantities)
        figures.plot_time(labels)
        figures.plot_traj(labels)
        plt.show()


if __name__ == '__main__':
    main()
