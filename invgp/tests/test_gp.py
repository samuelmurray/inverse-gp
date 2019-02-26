import unittest

import gpytorch
import numpy as np
import torch

from invgp.model import GP
from invgp.simulator import HeavySimulator


class TestExpectedImprovement(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1534315123)
        heavy_simulator = HeavySimulator()
        self.x = torch.Tensor(np.random.normal(size=[10, 2]))
        self.output_train = heavy_simulator(self.x)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GP(self.x, self.output_train, self.likelihood)

    def test_get_inputs(self) -> None:
        model_inputs = self.model.get_inputs()
        self.assertTrue(torch.equal(self.x, model_inputs))


if __name__ == "__main__":
    unittest.main()
