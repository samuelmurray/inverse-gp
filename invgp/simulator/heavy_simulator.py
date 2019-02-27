import numpy as np
import torch

from .simulator import Simulator


class HeavySimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input x should be a 2D tensor, but is {x.dim()}D")
        return torch.sum(torch.sin(x * (2 * np.pi)) + torch.randn(x.size()) * 0.2, dim=(1,))
