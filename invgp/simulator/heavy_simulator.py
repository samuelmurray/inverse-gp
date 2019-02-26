import numpy as np
import torch

from .simulator import Simulator


class HeavySimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sin(x * (2 * np.pi)) + torch.randn(x.size()) * 0.2, dim=(1,))
