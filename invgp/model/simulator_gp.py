import gpytorch
import torch

from invgp.simulator import Simulator
from .gp import GP


class SimulatorGP(GP):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood,
                 simulator: Simulator) -> None:
        super().__init__(x, y, likelihood)
        self.simulator = simulator

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        x_sim = self.simulator(x)
        mean_x = self.mean_module(x_sim)
        covariance_x = self.covariance_module(x_sim)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)
