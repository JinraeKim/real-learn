import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPRegression():
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        # initialise likelihood and model
        self.likelihood = GaussianLikelihood()

    def get_data(self, new_x, new_y):
        if len(self.train_x) == 0:
            self.train_x = new_x.unsqueeze(0)
            self.train_x = new_x.unsqueeze(0)
        else:
            self.train_x = torch.cat((self.train_x, new_x.unsqueeze(0)), 0)
        if len(self.train_y) == 0:
            self.train_y = new_y.unsqueeze(0)
            self.train_y = new_y.unsqueeze(0)
        else:
            self.train_y = torch.cat((self.train_y, new_y.unsqueeze(0)), 0)
    
    def train(self, lr=0.5, training_iter=50):
        self.model = ExactGPModel(self.train_x,
                                  self.train_y,
                                  self.likelihood)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {
                'params': self.model.parameters()
            },
        ], lr=lr)

        # 'Loss' for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calculate loss and backup gradients
            loss = -mll(output, self.train_y)
            loss.backward()  # update the gradients of each parameters
            # print((f'Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}  '
            #        + 'lengthscale: {lengthscale:.3f}  noise: {noise:.3f}'.format(
            #            lengthscale
            #            =self.model.covar_module.base_kernel.lengthscale.item(),
            #            noise=self.model.likelihood.noise.item())))
            optimizer.step()

    def eval(self, test_x):
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.model(test_x)
            observed_pred = self.likelihood(self.model(test_x))

        return pred, observed_pred


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
