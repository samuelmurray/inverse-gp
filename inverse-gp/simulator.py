import torch
import numpy as np


class Simulator:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.sin(x * (2 * np.pi)) + torch.randn(x.size()) * 0.2
