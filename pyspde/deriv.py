from math import sqrt

import numpy as np

from pyspde.lattice import Lattice
from pyspde.boundary import Boundary


class CentralDifference:

    def __init__(self, lattice):
        self._dx = lattice.increment

    def __call__(self, u):
        derivative = np.zeros(u.shape)
        # interior points
        derivative[:, 1:-1] = (u[:, 2:] - u[:, :-2])/(2*self._dx)
        # extrapolate boundary points
        derivative[:, 0] = 2*derivative[:, 1] - derivative[:, 2]
        derivative[:, -1] = 2*derivative[:, -2] - derivative[:, -3]
        return derivative


class Laplacian:

    def __init__(self, lattice, boundaries):
        self._dx = lattice.increment
        self._g = boundaries[1]

    def __call__(self, u):
        derivative = np.zeros(u.shape)
        # interior points
        derivative[:, 1:-1] = (u[:, 2:] -2*u[:, 1:-1] + u[:, :-2])/(self._dx**2)
        # extrapolate boundary at x = 0
        derivative[:, 0] = 2*derivative[:, 1] - derivative[:, 2]
        # set boundary at x = 1 according to derivative
        derivative[:, -1] = 2*(self._g(u[:, -1])/self._dx - (u[:, -1] - u[:, -2])/self._dx**2)
        return derivative


class BackwardDifference:

    def __init__(self, lattice):
        self._dx = lattice.increment

    def __call__(self, u):
        derivative = np.zeros(u.shape)
        # interior points
        derivative[:, 1:] = (u[:, 1:] - u[:, :-1])/self._dx
        # extrapolate boundary point
        derivative[:, 0] = 2*derivative[:, 1] - derivative[:, 2]
        return derivative
        

class LamShinDerivativeSquared:

    def __init__(self, lattice):
        self._dx = lattice.increment

    def __call__(self, u):
        # TODO: fix
        # compute square of derivative - inner points
        forward = u[:, 2:] - u[:, 1:-1]
        backward = u[:, 1:-1] - u[:, :-2]
        inner_points = forward**2 + forward*backward + backward**2
        derivative = np.zeros(u.shape)
        derivative[:, 1:-1] = inner_points
        # extrapolate boundary points
        derivative[:, 0] = 2*derivative[:, 1] - derivative[:, 2]
        derivative[:, -1] = 2*derivative[:, -2] - derivative[:, -3]
        return derivative/(3*self._dx**2)

