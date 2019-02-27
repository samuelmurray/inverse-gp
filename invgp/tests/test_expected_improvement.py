import unittest

import gpytorch
import numpy as np
import torch

from invgp.acquisition_function import ExpectedImprovement
from invgp.model import SimulatorGP
from invgp.simulator import HeavySimulator, SimpleSimulator


class TestExpectedImprovement(unittest.TestCase):
    """
    Tests for acquisition_function.expected_improvement.py
    """

    def setUp(self) -> None:
        """
        Reset numpy random seed
        :return:
        """
        np.random.seed(1534315123)

    def test_forward_return_shape(self) -> None:
        """
        Method should return a tensor with shape equal to number of candidate points
        :return:
        """
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
        self.assertEqual(torch.Size([num_candidates]), expected_improvement.shape)


if __name__ == "__main__":
    unittest.main()
