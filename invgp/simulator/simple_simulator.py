import torch

from simulator.simulator import Simulator


class SimpleSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
