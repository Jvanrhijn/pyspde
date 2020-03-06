from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.fftpack import dst, idst, dct, idct, fft, ifft
from scipy.linalg import expm
from scipy.integrate import quad

from src.spde import Boundary

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
        #fem = self.newton_iterate(fem, fem0)
        return (boundary.reshape(fem.shape) + self._basis.lattice_values(fem)).reshape(function.shape)

    def inverse_jacobian(self, function):
        gprime = self._gderiv(self._f + function.T @ self._phi_right)
        return -(np.eye(len(function)) \
            + gprime * self._qphi / (1 - gprime * self._phi_right @ self._q))

    def G(self, function):
        return (self._g(self._f \
            + function.T @ self._phi_right)*self._phi_right).reshape(function.shape)

    def fixed_point_iterate(self, x, v0, it_max=100, tolerance=1e-10):
        for it in range(it_max):
            x_old = x
            x = self._contract(x, v0)
            if np.linalg.norm(x - x_old) < tolerance:
                break
            if it == it_max-1:
                raise Warning("maximum iterations reached")
        return x

    def newton_iterate(self, x, v0, it_max=100, tolerance=1e-10):
        for it in range(it_max):
            x_old = x
            x = x - self.inverse_jacobian(x) @ (self._contract(x, v0) - x)
            residue = np.linalg.norm(x - x_old)
            if residue < tolerance:
                break
            if it == it_max-1:
                raise Warning("maximum Newton iterations reached")
        return x

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


class SpectralSolver(LinearSolver):

    def __init__(self, problem, store_midpoint=False):
        print("WARNING: spectral solver can only handle Dirichlet-Neumann boundaries")
        self._left = problem.left()
        self._right = problem.right
        self._gderiv = lambda u: (self._right(u + 0.001) - self._right(u - 0.001)) / 0.002
        points = len(problem.lattice.points)
        self._d = ((np.arange(1, points+1) - 0.5)*np.pi)**2
        self._d_inv = 1/self._d
        # TODO: generalize to L != 1
        self._xs = problem.lattice.points
        # midpoint value for theta
        self._store_midpoint = store_midpoint
        self._theta = None
        self._linear = problem.spde.linear

    def set_timestep(self, dt):
        self._dt = dt
        self._propagator = np.exp(-self._linear*self._d*dt)

    def propagate_step(self, u, problem):
        if self._linear == 0:
            return u

        max_iters = 100
        tolerance = 1e-10

        v0 = u - self._left - self._xs * self._right(u[:, -1])
        v0hat = dst(v0, type=3, norm='ortho')
        import matplotlib.pyplot as plt


        u_boundary = u[:, -1]
        ut_boundary = 0

        thetahat = np.zeros(u.shape)

        for it in range(max_iters):
            vhat = self._propagator * v0hat + (1 - self._propagator) * self._d_inv * thetahat
            dvhat_dt = -self._d * vhat + thetahat

            v = idst(vhat, type=3, norm='ortho')
            dvdt = idst(dvhat_dt, type=3, norm='ortho')

            u_boundary_old = u_boundary
            
            # newton iteration
            p = v[:, -1] + self._left
            u_boundary = u_boundary \
                - (p + self._xs[-1]*self._right(u_boundary) - u_boundary) / (self._xs[-1]*self._gderiv(u_boundary) - 1)

            ut_boundary = dvdt[:, -1] + self._xs[-1]*self._gderiv(u_boundary)*ut_boundary

            thetahat = -self._gderiv(u_boundary)*ut_boundary*dst(self._xs, type=3, norm='ortho')

            if abs(u_boundary - u_boundary_old) < tolerance:
                break

            if it == max_iters-1:
                print("WARNING: max iters reached")

        out = self._left + self._xs.T*self._right(u_boundary) + idst(vhat, type=3, norm='ortho')
        return out.reshape(u.shape)


class SpectralPeriodic(LinearSolver):

    def __init__(self, problem):
        points = len(problem.lattice.points)
        ns = np.arange(1, points+1)
        self._d = (ns * np.pi)**2
        self._propagator = None

    def set_timestep(self, dt):
        self._propagator = np.exp(-self._d * dt)
        self._dt = dt

    def propagate_step(self, u, problem):
        # transform to Fourier space
        uhat = fft(u)
        # propagate
        uhat *= self._propagator
        # transform back
        return np.abs(ifft(uhat))