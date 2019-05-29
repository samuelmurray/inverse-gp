from typing import Tuple

import numpy as np
import torch

from .simulator import Simulator


# TODO: The dimensions of matrices are not the same as expected in my other code.
#       It's expected that the forward method takes in a NxD matrix,
#       where N is the number of data points, and D is the dimension.
#       I.e. the calling code should not have the know about time steps etc.

class FlowSimulator(Simulator):
    def __init__(self, dt: float, nt: int) -> None:
        super().__init__()
        self.dt = dt
        self.nt = nt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_as_ndarray = x.numpy()
        _, fwd = self.rk4_solver_fwd(x_as_ndarray)
        norm = fwd[-1, 0] ** 2 + fwd[-1, 1] ** 2
        print(norm)
        fwd_as_tensor = torch.tensor([norm])
        print(fwd_as_tensor.shape)
        return fwd_as_tensor

    @staticmethod
    def rhs_fwd(x: np.ndarray) -> np.ndarray:
        """ RHS of the forward problem: dX/dt=f(X(t))
        :param x: d-by-1 vector, solution variable
        """
        rhs = np.zeros(len(x))
        rhs[0] = -x[0] + 10.0 * x[1]
        rhs[1] = x[1] * (10.0 * np.exp(-x[0] * x[0] / 100.0) - x[1]) * (x[1] - 1.0)
        return rhs

    def rk4_solver_fwd(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ RK4 solver for the forward problem
        :param x0: N-by-d vector, initial condition
        """
        t = np.zeros(self.nt)
        d = x0.shape[1]
        x = np.zeros((self.nt, d))
        x[0, :] = x0[:, 0]
        for i in range(1, self.nt):
            im = i - 1
            k1 = self.rhs_fwd(x[im, :])
            k2 = self.rhs_fwd(x[im, :] + 0.5 * self.dt * k1)
            k3 = self.rhs_fwd(x[im, :] + 0.5 * self.dt * k2)
            k4 = self.rhs_fwd(x[im, :] + self.dt * k3 + self.dt * k1)
            x[i, :] = x[im, :] + (self.dt / 6.0) * (k1 + 2. * k2 + 2. * k3 + k4)
            t[i] = t[im] + self.dt
        return t, x
