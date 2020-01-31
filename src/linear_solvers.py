from abc import ABC, abstractmethod
from functools import partial
from math import sin, cos, sqrt
import copy

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import norm


class LinearSolver(ABC):

    @abstractmethod
    def propagate_step(self, u):
        raise NotImplementedError()


class GalerkinSolver(LinearSolver):

    def __init__(self, spde, dt):
        """Store data and precompute lots of stuff"""
        self._xs = np.linspace(1/spde.points, 1, spde.points)
        self._spde = spde
        self._gderiv = lambda u: (spde.right(u + 0.001) - spde.right(u)) / 0.001
        # assume sine basis, we'll generalize later
        self._basis_spectral = [partial(self._basis_sine, n) for n in range(1, spde.points)]\
            + [lambda x: x]

        self._basis_spectral_deriv = [partial(self._basis_sine_deriv, n) for n in range(1, spde.points)]\
            + [lambda x: np.ones(x.shape) if type(x) == np.ndarray else 1]

        # precompute Galerkin matrices
        dim = spde.points
        self._a = np.zeros((dim, dim))
        self._b = np.zeros((dim, dim))
        phi = np.zeros((dim, dim))
        self._phi_right = np.array([self._basis_spectral[i](self._xs[-1]) for i in range(dim)])

        # build a manually
        ns = np.arange(0, spde.points, 1)
        k = GalerkinSolver.k
        np.fill_diagonal(self._a, 0.5)
        self._a[:-1, -1] = self._a[-1, :-1] = np.sin(k(ns[1:]))/k(ns[1:])**2
        self._a[-1, -1] = 1/3
        
        # build b manually
        np.fill_diagonal(self._b, 0.5*k(ns[1:])**2)
        self._b[:-1, -1] = self._b[-1, :-1] = np.sin(k(ns[1:]))
        self._b[-1, -1] = 1
        for i in range(dim):
            for j in range(dim):
                phi[i, j] = self._basis_spectral[j](self._xs[i])
        self._fem_to_sol = phi
        self._sol_to_fem = np.linalg.inv(phi)

        # more precomputation of useful quantities
        self._d = spde.linear * np.linalg.inv(self._a) @ self._b
        self._b_inv = np.linalg.inv(self._b)
        self._propagator = expm(-self._d*dt)

        self._s = (np.eye(dim) - self._propagator) @ self._b_inv
        self._q = self._s @ self._phi_right
        self._qphi = np.outer(self._q, self._phi_right)


    def propagate_step(self, u):
        v = self._sol_to_fem @ (u - self._spde.left)
        v0 = v
        v = self.newton_iterate(v, v0)
        return self._spde.left + self._fem_to_sol @ v

    def inverse_jacobian(self, u):
        gprime = self._gderiv(self._spde.left + u @ self._phi_right)
        return -(np.eye(self._spde.points) + gprime * self._qphi / (1 - gprime * self._phi_right @ self._q))

    def G(self, u): 
        return self._spde.right(self._spde.left + np.dot(u, self._phi_right))*self._phi_right

    def newton_iterate(self, x, v0, it_max=100, tolerance=1e-10):
        func = lambda y: self._contract(y, v0) - y
        for it in range(it_max):
            x_old = x
            x = x - self.inverse_jacobian(x) @ func(x)
            if np.linalg.norm(x - x_old) < tolerance:
                break
        return x

    def _contract(self, v, v0):
        prop = self._propagator
        return (np.eye(self._spde.points) - prop) @ self._b_inv @ self.G(v)\
            + prop @ v0

    @staticmethod
    def _basis_sine(n, x):
        k = GalerkinSolver.k
        return np.sin(k(n)*x)

    @staticmethod
    def _basis_sine_deriv(n, x):
        k = GalerkinSolver.k
        return k(n)*np.cos(k(n)*x)

    @staticmethod
    def k(n):
        return (n - 0.5)*np.pi
