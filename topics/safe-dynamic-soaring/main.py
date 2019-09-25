import numpy as np
from collections import deque

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from envs.soaring import Env


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class Agent:
    # Buffer size determines the number of points used to calculate
    # the derivatives
    buffer_size = 3
    sampling_freq = 10  # [Hz]

    def __init__(self, env, time_step):
        # self.env = env
        self.system = env.systems['aircraft']
        self.buffer = deque(maxlen=self.buffer_size)
        self.time_step = time_step
        self.sampling_interval = int(1 / time_step / self.sampling_freq)

        # GP setup
        self.model = ExactGPModel()

    def get_data(self):
        x, u = np.split(np.array(self.buffer), [6, ], axis=1)
        med_x, med_u = np.median(x, axis=0), np.median(u, axis=0)
        state_deriv = (x[-1] - x[0]) / (self.time_step * (self.buffer_size - 1))

        # Find f(x, u)
        f = self.system._raw_deriv(
            med_x, 0, med_u, {'wind': (np.zeros(3), np.zeros((3, 3)))})

        # Calculate x_dot - f(x, u) = h(x)*d(x)
        # where d(x) = (Wy_hat(x), dWydz_hat(x))
        hd = state_deriv - f

        Wy_hat = hd[1]

        *_, z, V, gamma, psi = med_x
        dWydz_hat = hd[3] / np.cos(gamma) / np.sin(gamma) / np.sin(psi) / V

        return z, (Wy_hat, dWydz_hat)


np.random.seed(1)

time_step = 0.005
time_series = np.arange(0, 2, time_step)

env = Env(
    initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
    dt=time_step
)
agent = Agent(env, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))

for i, t in enumerate(time_series):
    controls = [0, 1, 0]

    next_obs, reward, done, info = env.step(controls)

    agent.buffer.append(np.hstack((obs, controls)))

    if len(agent.buffer) == agent.buffer.maxlen \
       and i % agent.sampling_interval == 0:
        data, label = agent.get_data()

    if done:
        break

    obs = next_obs
    obs_series = np.vstack((obs_series, obs))
