from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from src.spde import Boundary


class BasisSet(ABC):

    @abstractmethod
    def member(self, index):
        pass

    @abstractmethod
    def __call__(self, index, *args):
        pass

    @abstractmethod
    def lattice_values(self, coefficients):
        pass

    @abstractmethod
    def coefficients(self, lattice_values):
        pass

    @abstractmethod
    def stiffness(self):
        pass

    @abstractmethod
    def mass(self):
        pass


class FiniteElementBasis(BasisSet):

    def __init__(self, lattice, boundaries):
        self._dimension = len(lattice.points)
        self._dx = lattice.increment
        self._functions = [partial(self._basis_hat, n) for n in range(self._dimension)]
        self._derivatives = [partial(self._basis_hat_deriv, n) for n in range(self._dimension)]
        # compute stiffness matrix
        self._stiffness = 1/self._dx * (np.diag(np.ones(self._dimension)*2) \
           - np.diag(np.ones(self._dimension-1), k=1)\
           - np.diag(np.ones(self._dimension-1), k=-1))
        self._stiffness[-1, -1] = 1/self._dx if boundaries[1].kind() == Boundary.ROBIN else 2/self._dx
        # compute mass matrix
        self._mass = (np.diag(np.ones(self._dimension)*2/3) \
            + np.diag(np.ones(self._dimension-1)*1/6, k=1)\
            + np.diag(np.ones(self._dimension-1)*1/6, k=-1))*self._dx
        self._mass[-1, -1] = self._dx/3 if boundaries[1].kind() == Boundary.ROBIN else 2*self._dx/3

    def member(self, index):
        return self._functions[index]

    def __call__(self, x, derivative=False):
        if not derivative:
            return np.array([fun(x) for fun in self._functions])
        else:
            return np.array([fun(x) for fun in self._derivatives])

    def stiffness(self):
        return self._stiffness

    def mass(self):
        return self._mass

    def lattice_values(self, coefficients):
        # FEM transformation is identity
        return coefficients

    def coefficients(self, coefficients):
        # FEM transformation is identity
        return coefficients

    def _basis_hat(self, n, x):
        dx = self._dx
        return np.piecewise(x,
                [
                    np.logical_and((n)*dx < x, x <= (n+1)*dx),
                    np.logical_and((n+1)*dx < x, x < (n+2)*dx),
                ],
                [
                    lambda x: (x - ((n)*dx))/dx,
                    lambda x: ((n+2)*dx - x)/dx,
                    lambda x: 0
                ])

    def _basis_hat_deriv(self, n, x):
        dx = self._dx
        return np.piecewise(x,
                [
                    np.logical_and((n)*dx <= x, x <= (n+1)*dx),
                    np.logical_and((n+1)*dx <= x, x <= (n+2)*dx),
                ],
                [
                    lambda x: 1/dx,
                    lambda x: -1/dx,
                    lambda x: 0
                ])