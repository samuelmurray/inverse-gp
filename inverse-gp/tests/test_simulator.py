import unittest

import torch

from simulators import HeavySimulator


class TestSimulator(unittest.TestCase):
    def test_call(self) -> None:
        simulator = HeavySimulator()
        x = torch.linspace(0, 1, 20)
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)
