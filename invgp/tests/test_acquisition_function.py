import unittest

from invgp.acquisition_function import AcquisitionFunction


class TestAcquisitionFunction(unittest.TestCase):
    """
    Tests for acquisition_function.acquisition_function.py
    """

    def test_abc(self) -> None:
        """
        Instantiation of AcquisitionFunction should not be possible
        :return:
        """
        self.assertRaises(TypeError, AcquisitionFunction)


if __name__ == "__main__":
    unittest.main()
