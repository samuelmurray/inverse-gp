import unittest

import gpytorch
import numpy as np
import torch

from invgp.acquisition_function import ExpectedImprovement
from invgp.model import SimulatorGP
from invgp.simulator import HeavySimulator, SimpleSimulator


class TestExpectedImprovement(unittest.TestCase):
    def test_call(self) -> None:
        np.random.seed(1534315123)
        simple_simulator = SimpleSimulator()
        heavy_simulator = HeavySimulator()
        num_data = 10
        input_dim = 2
        x = torch.Tensor(np.random.normal(size=[num_data, input_dim]))
        y = heavy_simulator(x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SimulatorGP(x, y, likelihood, simple_simulator)
        acquisition_function = ExpectedImprovement(model)
        num_candidates = 100
        candidate_set = torch.Tensor(np.random.normal(size=[num_candidates, input_dim]))
        expected_improvement = acquisition_function(x, y, candidate_set)
        self.assertEqual(np.array([num_candidates]), expected_improvement.shape)


if __name__ == "__main__":
    unittest.main()
