import numpy as np


class DerivativeOperator:

    def __init__(self, order, dx, left, right):
        self._order = order
        self._dx = dx
        self._left = left
        self._right = right

    def __call__(self, u):
        # TODO Improve this
        dim = len(u)
        matrix = np.zeros((dim, dim))
        if self._order == 1:
            derivative = np.zeros(dim)
            derivative[1:-1] = (u[2:] - u[:-2])/(2*self._dx)
            derivative[0] = derivative[1]
            derivative[-1] = self._right(u[-1])
        elif self._order == 2:
            raise NotImplementedError("Second-order FD not yet implemented")
            derivative = np.zeros(dim)
            derivative[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/self._dx**2
            derivative[0] = derivative[1]
            derivative[-1] = 2 * (self._right(u[-1])/self._dx -
                                  (u[-1] - u[-2])/self._dx**2)
        return derivative