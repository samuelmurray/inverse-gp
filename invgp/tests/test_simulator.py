import unittest

import numpy as np
import torch

from invgp.simulator import *


class TestSimulator(unittest.TestCase):
    def test_abc(self) -> None:
        self.assertRaises(TypeError, Simulator)


class TestHeavySimulator(unittest.TestCase):
    def test_call(self) -> None:
        simulator = HeavySimulator()
        x = torch.Tensor(np.random.normal(size=[10, 2]))
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


class TestSimpleSimulator(unittest.TestCase):
    def test_call(self) -> None:
        simulator = SimpleSimulator()
        x = torch.Tensor(np.random.normal(size=[10, 2]))
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
