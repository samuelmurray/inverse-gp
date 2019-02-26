import torch

from simulator import Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
