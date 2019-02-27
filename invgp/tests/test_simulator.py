import unittest

import numpy as np
import torch

from invgp.simulator import *


class TestSimulator(unittest.TestCase):
    """
    Tests for simulator.simulator.py
    """

    def test_abc(self) -> None:
        """
        Instantiation of Simulator should not be possible
        :return:
        """
        self.assertRaises(TypeError, Simulator)


class TestHeavySimulator(unittest.TestCase):
    """
    Tests for simulator.heavy_simulator.py
    """

    def setUp(self):
        """
        Reset numpy random seed
        :return:
        """
        np.random.seed(1534315123)

    def test_call(self) -> None:
        """
        Returned Tensor should be as long as number of datapoints
        :return:
        """
        num_data = 10
        num_dim = 2
        simulator = HeavySimulator()
        x = torch.Tensor(np.random.normal(size=[num_data, num_dim]))
        y = simulator(x)
        self.assertEqual(num_data, y.shape[0])


class TestSimpleSimulator(unittest.TestCase):
    """
    Tests for simulator.simple_simulator.py
    """

    def setUp(self):
        """
        Reset numpy random seed
        :return:
        """
        np.random.seed(1534315123)

    def test_call(self) -> None:
        """
        Returned Tensor should be as long as number of datapoints
        :return:
        """
        num_data = 10
        num_dim = 2
        simulator = SimpleSimulator()
        x = torch.Tensor(np.random.normal(size=[num_data, num_dim]))
        y = simulator(x)
        self.assertEqual(num_data, y.shape[0])


if __name__ == "__main__":
    unittest.main()
