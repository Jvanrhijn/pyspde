import numpy as np

from src.spde import Lattice


class DerivativeOperator:

    def __init__(self, order, lattice, boundaries):
        self._order = order
        self._dx = lattice.increment
        # TODO: implement different boundaries
        self._left = boundaries[0]
        self._right = boundaries[1]

    def __call__(self, u):
        # TODO: fix shape
        u = u.reshape((1, u.size))
        if self._order == 1:
            derivative = np.zeros(u.shape)
            derivative[:, 1:-1] = (u[:, 2:] - u[:, :-2])/(2*self._dx)
            derivative[:, 0] = derivative[:, 1]
            derivative[:, -1] = self._right(u[:, -1])
        elif self._order == 2:
            raise NotImplementedError("Second-order FD not yet implemented")
            derivative = np.zeros(u.shape)
            derivative[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2])/self._dx**2
            derivative[:, 0] = derivative[:, 1]
            derivative[:, -1] = 2 * (self._right(u[:, -1])/self._dx -
                                  (u[:, -1] - u[:, -2])/self._dx**2)
        return derivative.reshape(u.shape)