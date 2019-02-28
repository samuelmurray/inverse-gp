import unittest

import numpy as np
import torch

from invgp.simulator import SimpleSimulator


class TestSimpleSimulator(unittest.TestCase):
    """
    Tests for simulator.simple_simulator.py
    """

    def setUp(self) -> None:
        """
        Reset numpy random seed
        :return:
        """
        np.random.seed(1534315123)

    def test_forward_raises_on_1D_input(self):
        """
        Only input tensors that are 2D are valid
        :return:
        """
        simulator = SimpleSimulator()
        x = torch.linspace(1, 10)
        self.assertRaises(ValueError, simulator, x)

    def test_forward_return_shape(self) -> None:
        """
        Returned Tensor should be as long as number of datapoints
        :return:
        """
        num_data = 10
        num_dim = 2
        simulator = SimpleSimulator()
        x = torch.as_tensor(np.random.normal(size=[num_data, num_dim]), dtype=torch.float32)
        y = simulator(x)
        self.assertEqual(num_data, y.shape[0])


if __name__ == "__main__":
    unittest.main()
