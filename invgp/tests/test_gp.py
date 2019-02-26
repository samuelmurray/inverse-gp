import unittest

import gpytorch
import numpy as np
import torch

from invgp.model import GP


class TestExpectedImprovement(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.Tensor(np.random.normal(size=[10, 2]))
        self.y = self.x.sin()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GP(self.x, self.y, self.likelihood)

    def test_get_inputs(self) -> None:
        model_inputs = self.model.get_inputs()
        self.assertTrue(self.x.equal(model_inputs))


if __name__ == "__main__":
    unittest.main()
