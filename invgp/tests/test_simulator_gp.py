import unittest

import gpytorch
import numpy as np
import torch

from invgp.model import SimulatorGP
from invgp.simulator import HeavySimulator, SimpleSimulator


class TestSimulatorGP(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1534315123)
        simple_simulator = SimpleSimulator()
        heavy_simulator = HeavySimulator()
        num_train = 10
        self.input_train = torch.Tensor(np.random.normal(size=[num_train, 2]))
        self.output_train = heavy_simulator(self.input_train)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SimulatorGP(self.input_train, self.output_train, self.likelihood, simple_simulator)

    def test_get_inputs(self) -> None:
        model_inputs = self.model.get_inputs()
        self.assertTrue(torch.equal(self.input_train, model_inputs))

    def test_predict(self) -> None:
        with torch.no_grad():
            self.model.eval()
            num_test = 5
            input_test = torch.Tensor(np.random.normal(size=[num_test, 2]))
            predictions = self.model(input_test).mean
            self.assertEqual(np.array([num_test]), predictions.shape)


if __name__ == "__main__":
    unittest.main()
