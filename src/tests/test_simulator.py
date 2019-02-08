import unittest

from simulator import Simulator


class TestSimulator(unittest.TestCase):
    def test_call(self):
        simulator = Simulator()
        x = 5.
        y = simulator(x)
        self.assertEqual(x, y)
