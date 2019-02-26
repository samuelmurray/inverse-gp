import unittest

import gpytorch
import torch

from invgp.model import GP


class TestExpectedImprovement(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.linspace(0, 1, 20).unsqueeze(0)
        self.y = self.x.sin()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GP(self.x, self.y, likelihood)

    def test_(self) -> None:
        model_inputs = self.model.get_inputs()
        self.assertTrue(self.x.equal(model_inputs))


if __name__ == "__main__":
    unittest.main()
