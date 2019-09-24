import numpy as np

from envs.soaring import Env

np.random.seed(1)

env = Env(initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'))

time_step = 0.01
time_series = np.arange(0, 2, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

for i in time_series:
    controls = [1, 1, 0]

    # Need logging here
    next_obs, reward, done, _ = env.step(controls)

    if done:
        break

    obs = next_obs
    obs_series = np.vstack((obs_series, obs))

print(i)
