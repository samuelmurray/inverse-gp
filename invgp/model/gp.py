import gpytorch
import torch


class GP(gpytorch.models.ExactGP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood) -> None:
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covariance_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def get_inputs(self) -> torch.Tensor:
        return self.train_inputs[0]

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        mean_x = self.mean_module(x)
        covariance_x = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)
