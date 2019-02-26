import torch

from simulator import Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x
