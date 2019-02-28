import unittest

import gpytorch
import numpy as np
import torch

from invgp.model import SimulatorGP
from invgp.simulator import HeavySimulator, SimpleSimulator


class TestSimulatorGP(unittest.TestCase):
    """
    Tests for model.simulator_gp.py
    """

    def setUp(self) -> None:
        """
        Reset numpy random seed, create training data and model
        :return:
        """
        np.random.seed(1534315123)
        torch.manual_seed(163153413413512)
        simple_simulator = SimpleSimulator()
        heavy_simulator = HeavySimulator()
        num_train = 10
        self.input_dim = 2
        self.input_train = torch.as_tensor(np.random.normal(size=[num_train, self.input_dim]), dtype=torch.float32)
        self.output_train = heavy_simulator(self.input_train)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SimulatorGP(self.input_train, self.output_train, self.likelihood, simple_simulator)

    def test_forward_raises_on_1D_input(self):
        """
        Only input tensors that are 2D are valid
        :return:
        """
        input_train = torch.linspace(0, 10)
        output_train = torch.linspace(0, 10)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        simulator = SimpleSimulator
        self.assertRaises(ValueError, SimulatorGP, input_train, output_train, likelihood, simulator)

    def test_get_inputs(self) -> None:
        """
        Method get_inputs() should return the training data used when instantiating model
        :return:
        """
        model_inputs = self.model.get_inputs()
        self.assertTrue(torch.equal(self.input_train, model_inputs))

    def test_forward_return_shape(self) -> None:
        """
        Method predict() should return a Distribution, whose mean has shape equal to number of test points
        :return:
        """
        with torch.no_grad():
            self.model.eval()
            num_test = 5
            input_test = torch.as_tensor(np.random.normal(size=[num_test, self.input_dim]), dtype=torch.float32)
            predictions = self.model(input_test).mean
            self.assertEqual(torch.Size([num_test]), predictions.shape)


if __name__ == "__main__":
    unittest.main()
