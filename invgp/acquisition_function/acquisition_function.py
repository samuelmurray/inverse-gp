import abc

import gpytorch
import torch

from invgp.model import GP


class AcquisitionFunction(gpytorch.Module, abc.ABC):
    def __init__(self, model: GP) -> None:
        super().__init__()
        self.model = model

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, candidate_set: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
