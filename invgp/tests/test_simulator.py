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


if __name__ == "__main__":
    unittest.main()
