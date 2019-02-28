import torch
from torch.distributions.normal import Normal

from .acquisition_function import AcquisitionFunction
from invgp.model import GP


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, model: GP) -> None:
        super().__init__(model)

    def forward(self, x: torch.Tensor, y: torch.Tensor, candidate_set: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.likelihood.eval()
        self.model.set_train_data(x, y, strict=False)

        pred = self.model.likelihood(self.model(candidate_set))
        mu = pred.mean.detach()
        var = pred.variance.detach()
        std = torch.sqrt(var)
        best_y = torch.max(y)

        u = (best_y - mu) / std
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        ucdf = m.cdf(u)
        updf = torch.exp(m.log_prob(u))
        expected_improvement = var * (updf + u * ucdf)
        return expected_improvement
