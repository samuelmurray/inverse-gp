import gpytorch
import torch
from torch.distributions.normal import Normal

from gp import GP


class Optimizer(gpytorch.Module):
    def __init__(self, model: GP, simulator):
        super().__init__()
        self.model = model
        self.simulator = simulator
        self.grid_size = 100

    def forward(self, x, y, candidate_set):
        self.model.eval()
        self.model.likelihood.eval()
        self.model.train_inputs = x
        self.model.train_targets = y

        pred = self.gp_model.likelihood(self.model(candidate_set))

        mu = pred.mean().detach()
        sigma = pred.std().detach()

        u = (self.best_y - mu) / sigma
        m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        ucdf = m.cdf(u)
        updf = torch.exp(m.log_prob(u))
        expected_improvement = sigma * (updf + u * ucdf)
        return expected_improvement
