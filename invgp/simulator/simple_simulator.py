import torch

from .simulator import Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Input x should be a 2D tensor, but is {x.dim()}D")
        return x
