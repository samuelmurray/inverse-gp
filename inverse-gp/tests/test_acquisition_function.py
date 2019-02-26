import unittest

import gpytorch
import numpy as np
import torch

from acquisition_functions import ExpectedImprovement
from simulators import SimpleSimulator
from model import GP


class TestExpectedImprovement(unittest.TestCase):
    def test_call(self) -> None:
        simulator = SimpleSimulator()
        x = torch.linspace(0, 1, 20)
        y = simulator(x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(x, y, likelihood)
        acquisition_function = ExpectedImprovement(model, simulator)
        num_candidates = 100
        candidate_set = torch.linspace(-1, 2, num_candidates)
        expected_improvement = acquisition_function(x, y, candidate_set)
        self.assertEqual(np.array([num_candidates]), expected_improvement.shape)


if __name__ == "__main__":
    unittest.main()
