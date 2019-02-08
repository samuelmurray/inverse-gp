import unittest

import torch

from simulator import Simulator


class TestSimulator(unittest.TestCase):
    def test_call(self):
        simulator = Simulator()
        x = torch.linspace(0, 1, 20)
        y = simulator(x)
        self.assertEqual(x.shape, y.shape)
