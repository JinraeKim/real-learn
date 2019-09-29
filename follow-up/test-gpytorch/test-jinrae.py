import math
from matplotlib import pyplot as plt

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood

# training data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# initialise likelihood and model
likelihood = GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# find optimal model hyperparameters
model.train()
likelihood.train()

# use the adam optimiser
optimiser = Adam([
    {
        'params': model.parameters()
    }
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    # zero gradients from previous iteration
    optimiser.zero_grad()
    # output from model
    output = model(train_x)
    # calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f    lengthscale: %.3f    noise: %.3f' %
         i + 1, training_iter, loss.item(),
         model.covar_module.base_kernel.lengthscale.item(),
         model.likelihood.noise.item()
         )
    optimiser.step()
