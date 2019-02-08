import gpytorch
import torch
from torch.distributions.normal import Normal
import numpy as np

from gp import GP


class AcquisitionFunction(gpytorch.Module):
    def __init__(self, model: GP, simulator):
        super().__init__()
        self.model = model
        self.simulator = simulator
        self.grid_size = 100

    def forward(self, x, y, candidate_set):
        self.model.eval()
        self.model.likelihood.eval()
        #self.model.train_inputs = x
        #self.model.set_train_data(x, y)
        model = self.model.get_fantasy_model(x, y)
        #self.model.train_targets = y

        pred = self.model.likelihood(self.model(candidate_set))

        mu = pred.mean.detach()
        var = pred.variance.detach()
        std = np.sqrt(var)

        best_y = torch.max(y)

        u = (best_y - mu) / std
        m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        ucdf = m.cdf(u)
        updf = torch.exp(m.log_prob(u))
        expected_improvement = var * (updf + u * ucdf)
        return expected_improvement
