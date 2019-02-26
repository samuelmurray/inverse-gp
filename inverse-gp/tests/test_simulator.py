import unittest

import torch

from simulators import *


class TestSimulator(unittest.TestCase):
    def test_abc(self) -> None:
        self.assertRaises(TypeError, Simulator)


class TestHeavySimulator(unittest.TestCase):
    def test_call(self) -> None:
        simulator = HeavySimulator()
        x = torch.linspace(0, 1, 20)
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


class TestSimpleSimulator(unittest.TestCase):
    def test_call(self) -> None:
        simulator = SimpleSimulator()
        x = torch.linspace(0, 1, 20)
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
