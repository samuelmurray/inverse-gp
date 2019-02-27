"""
Package for all acquisition functions
"""

__all__ = [
    "AcquisitionFunction",
    "ExpectedImprovement",
]

from .acquisition_function import AcquisitionFunction
from .expected_improvement import ExpectedImprovement
