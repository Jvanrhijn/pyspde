from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.fftpack import dst, idst, dct, idct, fft, ifft
from scipy.linalg import expm
from scipy.integrate import quad

from pyspde.boundary import Boundary


class LinearSolver(ABC):

    @abstractmethod
    def propagate_step(self, function):
        raise NotImplementedError()


class GalerkinSolver(LinearSolver):

    def __init__(self, problem, basis):
        #raise NotImplementedError("Galerkin method not yet implemented")
        """Store data and precompute lots of stuff"""
        self._range = problem.lattice.range
        self._xs = problem.lattice.points
        self._f = problem.left()
        self._g = problem.right
        self._gderiv = lambda u: (self._g(u + 0.001) - self._g(u - 0.001))/0.002

        self._basis = basis

        # precompute Galerkin matrices
        self._b = self._basis.stiffness()
        self._a = self._basis.mass()
        self._phi = self._basis(self._xs)
        self._phi_right = self._phi[:, -1]

        self._fem_to_sol = self._phi
        self._sol_to_fem = np.linalg.inv(self._phi)

        # more precomputation of useful quantities
        self._d = problem.spde.linear * np.linalg.inv(self._a) @ self._b
        self._b_inv = np.linalg.inv(self._b)

    def set_timestep(self, dt):
        self._dt = dt
        self._propagator = expm(-self._d*dt)
        dim = len(self._d)

        self._s = (np.eye(dim) - self._propagator) @ self._b_inv
        self._q = self._s @ self._phi_right
        self._qphi = np.outer(self._q, self._phi_right)

    def propagate_step(self, function, problem):
        if problem.spde.linear == 0:
            # do nothing if the linear part is zero
            return function 
        boundary = self.get_boundary(function, self._xs, problem)[0].reshape(function.shape)
        fem = self._basis.coefficients(function.T - boundary.T)
        fem0 = fem
        fem = self._contract(fem, fem0)
        return (boundary.reshape(fem.shape) + self._basis.lattice_values(fem)).reshape(function.shape)

    def G(self, function):
        return (self._g(self._f \
            + function.T @ self._phi_right)*self._phi_right).reshape(function.shape)

    def _contract(self, v, v0):
        prop = self._propagator
        return (np.eye(len(v)) - prop) @ self._b_inv @ self.G(v)\
            + prop @ v0

    def get_boundary(self, field, xs, problem):
        delta = problem.lattice.range[1] - problem.lattice.range[0]
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return problem.left() * np.ones(field.shape), 0
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            slope = (problem.right() - problem.left())/delta
            return problem.left() + slope*(xs - problem.lattice.range[0]), slope
        else:
            raise NotImplementedError("Boundary combination not yet implemented")