import abc

import gpytorch
import torch


class Simulator(gpytorch.Module, abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
