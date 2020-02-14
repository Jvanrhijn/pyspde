from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.integrate import quad

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
        self._origin, self._end = lattice.range
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

    def coefficients(self, lattice_values):
        # FEM transformation is identity
        return lattice_values

    def coefficients_generic(self, values, xs):
        # for any lattice points, we can generate the matrix:
        phis = self(xs)
        # take the first N columns and transform with that
        return (np.linalg.inv(phis[:, :-1].T) @ values[:, :-1].T).reshape((1, xs.size-1))


    def _basis_hat(self, n, x):
        dx = self._dx
        x0 = self._origin
        x = x - x0
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
        x0 = self._origin
        x = x - x0
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


class SpectralBasis(BasisSet):

    def __init__(self, lattice, boundaries):
        #raise NotImplementedError("Spectral basis not functional yet")
        points = len(lattice.points)
        xs = lattice.points
        self._k = lambda n: (n-0.5)*np.pi
        self._functions = [partial(self._basis_sine, n) for n in range(1, points)] \
            + [lambda x: x]
        self._derivatives = [partial(self._basis_sine_deriv, n) for n in range(1, points)] \
            + [lambda x: np.ones(x.shape) if isinstance(x, np.ndarray) else 1] 
        self._mass = np.zeros((points, points))
        self._stiffness = np.zeros((points, points))
        self._fem_to_sol = np.zeros((points, points))
        # matrices for converting between coefficients and solution
        for i in range(points):
            for j in range(points):
                self._fem_to_sol[i, j] = self._functions[j](xs[i])
                self._mass[i, j] = quad(lambda x: self._functions[j](x)*self._functions[i](x), 0, 1)[0]
                self._stiffness[i, j] = quad(lambda x: self._derivatives[j](x)*self._derivatives[i](x), 0, 1)[0]
        self._sol_to_fem = np.linalg.inv(self._fem_to_sol)

    def __call__(self, x, derivative=False):
        if not derivative:
            return np.array([fun(x) for fun in self._functions])
        else:
            return np.array([fun(x) for fun in self._derivatives])

    def coefficients(self, lattice_points):
        return (self._sol_to_fem @ lattice_points.T).reshape(lattice_points.shape)

    def lattice_values(self, coefficients):
        return (self._fem_to_sol @ coefficients.T).reshape(coefficients.shape)

    def member(self, index):
        return self._functions[index]

    def stiffness(self):
        return self._stiffness

    def mass(self):
        return self._mass

    def _basis_sine(self, n, x):
        return np.sin(self._k(n)*x)

    def _basis_sine_deriv(self, n, x):
        return self._k(n)*np.cos(self._k(n)*x)