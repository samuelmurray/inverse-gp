import unittest

import numpy as np
import torch

from invgp.simulator import *


class TestSimulator(unittest.TestCase):
    def test_abc(self) -> None:
        self.assertRaises(TypeError, Simulator)


class TestHeavySimulator(unittest.TestCase):
    def setUp(self):
        np.random.seed(1534315123)

    def test_call(self) -> None:
        num_data = 10
        simulator = HeavySimulator()
        x = torch.Tensor(np.random.normal(size=[num_data, 2]))
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


class TestSimpleSimulator(unittest.TestCase):
    def setUp(self):
        np.random.seed(1534315123)

    def test_call(self) -> None:
        num_data = 10
        simulator = SimpleSimulator()
        x = torch.Tensor(np.random.normal(size=[num_data, 2]))
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
