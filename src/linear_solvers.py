from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.fftpack import dst, idst, dct, idct
from scipy.linalg import expm
from scipy.integrate import quad


class LinearSolver(ABC):

    @abstractmethod
    def propagate_step(self, function):
        raise NotImplementedError()


class GalerkinSolver(LinearSolver):

    def __init__(self, spde):
        """Store data and precompute lots of stuff"""
        self._range = spde.space_range
        self._xs = np.linspace(self._range[0] + 1/spde.points, self._range[1], spde.points)
        self._spde = spde
        self._gderiv = spde.right_deriv


        # assume sine basis, we'll generalize later
        self._basis_spectral = [partial(self._basis_sine, n)
                                for n in range(1, spde.points)]\
            + [lambda x: x - self._range[0]]

        self._basis_spectral_deriv = [partial(self._basis_sine_deriv, n)
                                      for n in range(1, spde.points)]\
            + [lambda x: np.ones(x.shape) if isinstance(x, np.ndarray) else 1]

        # precompute Galerkin matrices
        dim = spde.points
        self._a = np.zeros((dim, dim))
        self._b = np.zeros((dim, dim))
        phi = np.zeros((dim, dim))
        self._phi_right = np.array(
            [self._basis_spectral[i](self._xs[-1]) for i in range(dim)])

        # build a manually
        ns = np.arange(0, spde.points, 1)
        k = self.k
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

    def set_timestep(self, dt):
        self._dt = dt
        self._propagator = expm(-self._d*dt)
        dim = len(self._d)

        self._s = (np.eye(dim) - self._propagator) @ self._b_inv
        self._q = self._s @ self._phi_right
        self._qphi = np.outer(self._q, self._phi_right)

    def propagate_step(self, function):
        fields = function.shape[0]
        if self._spde.linear == 0:
            # do nothing if the linear part is zero
            return function 
        propagated_field = np.zeros(function.shape)
        for field in range(fields):
            fem = self._sol_to_fem @ (function[field] - self._spde.left[field])
            fem0 = fem
            fem = self.newton_iterate(fem, fem0, field)
            propagated_field[field] = self._spde.left[field] + self._fem_to_sol @ fem
        return propagated_field
        #return self._spde.left + self._fem_to_sol @ fem

    def inverse_jacobian(self, function, field_index):
        gprime = self._gderiv[field_index](self._spde.left[field_index] + function @ self._phi_right)
        return -(np.eye(self._spde.points) \
            + gprime * self._qphi / (1 - gprime * self._phi_right @ self._q))

    def G(self, function, field_index):
        return self._spde.right[field_index](self._spde.left[field_index] \
            + function @ self._phi_right)*self._phi_right

    def fixed_point_iterate(self, x, v0, field_index, it_max=100, tolerance=1e-10):
        for it in range(it_max):
            x_old = x
            x = self._contract(x, v0, field_index)
            if np.linalg.norm(x - x_old) < tolerance:
                break
            if it == it_max-1:
                raise Warning("maximum Newton iterations reached")
        return x

    def newton_iterate(self, x, v0, field_index, it_max=100, tolerance=1e-10):
        for it in range(it_max):
            x_old = x
            x = x - self.inverse_jacobian(x, field_index) @ (self._contract(x, v0, field_index) - x)
            if np.linalg.norm(x - x_old) < tolerance:
                break
            if it == it_max-1:
                raise Warning("maximum Newton iterations reached")
        return x

    def _contract(self, v, v0, field_index):
        prop = self._propagator
        return (np.eye(self._spde.points) - prop) @ self._b_inv @ self.G(v, field_index)\
            + prop @ v0

    def _basis_sine(self, n, x):
        k = self.k
        return np.sin(k(n)*(x - self._range[0]))

    def _basis_sine_deriv(self, n, x):
        k = self.k(n)
        return k*np.cos(k*(x - self._range[0]))

    def k(self, n):
        return (n - 0.5)*np.pi / (self._range[1] - self._range[0])


class SpectralSolver(LinearSolver):

    def __init__(self, spde, store_midpoint=False):
        self._spde = spde
        self._transform = Transform(
            [1, -1], spde.left, spde.right, spde.points)
        self._d = self._transform.derivative_matrix
        self._d_inv = 1/self._d
        # TODO: generalize to L != 1
        self._xs = np.linspace(1/spde.points, 1, spde.points)
        # midpoint value for theta
        self._store_midpoint = store_midpoint
        self._theta = None

    def set_timestep(self, dt):
        self._dt = dt
        self._propagator = np.exp(-self._spde.linear*self._d*dt)

    def propagate_step(self, u):
        fields = u.shape[0]
        if self._spde.linear == 0:
            return u
        out = np.zeros(u.shape)
        
        max_iters = 100
        tolerance = 1e-10

        # If we have a stored midpoint theta, use that
        theta = self._theta if self._theta is not None else np.zeros(u.shape)

        for field in range(fields):
            v0 = self._transform.homogenize(u[field], self._xs, field)
            v0hat = self._transform.fft(v0)

            u_boundary = u[field][-1]
            ut_boundary = 0

            for it1 in range(max_iters):
                # Compute homogenized function in Fourier space using IP
                vhat = (1 - self._propagator) * self._d_inv * theta[field] + self._propagator * v0hat
                # Compute time-derivative in Fourier space using the ODE for vhat
                dvhat_dt = -self._d * vhat + theta[field]

                # Transform back to real space
                v = self._transform.ifft(vhat)
                dvdt = self._transform.ifft(dvhat_dt)

                u_boundary_old = u_boundary
                u_boundary, ut_boundary = self._transform.boundary_iteration(
                    self._xs, v, dvdt, u_boundary, ut_boundary, field)

                # only compute new theta iterate if we don't have a midpoint value
                # stored
                if self._theta is None:
                    # Compute new value for theta using new boundary values for u
                    theta[field] = self._transform.fft(self._transform.theta(
                        self._xs, u_boundary, ut_boundary, field))

                # check convergence
                if abs(u_boundary - u_boundary_old) < tolerance:
                    break

                if it1 == max_iters-1:
                    print("WARNING: outer loop max iters reached")


            out[field] = self._transform.dehomogenize(v, self._xs, u_boundary, field)

        # store midpoint theta if needed, else delete it
        self._theta = theta if self._theta is None and self._store_midpoint else None

        # return solution value at this time point
        return out


class Transform:
    """
    This class contains all logic required to perform an interaction picture
    step. It contains code to trsform the function to homogeneous boundaries,
    to compute the inhomogeneous function $\theta$, and to compute
    one fixed-point iteration of the boundary values.
    """

    def __init__(self, boundaries, f, g, points, gderiv=None):
        """
        Parameters
        ----------
        boundaries: list
            List of boundary types. A '1' means Dirichlet, '-1' means Robin
        f: float
            Fixed boundary value
        g: float or function
            Fixed boundary value if boundaries == [1, 1], or Robin
            boundary function otherwise
        """
        self._boundaries = boundaries
        self._f = f
        self._g = g
        if boundaries == [1, 1]:
            self.fft = lambda s, **kwargs: dst(s, norm='ortho', type=1)
            self.ifft = lambda s, **kwargs: idst(s, norm='ortho', type=1)
        elif boundaries == [1, -1]:
            self.fft = lambda s, **kwargs: dst(s, norm='ortho', type=3)
            self.ifft = lambda s, **kwargs: idst(s, norm='ortho', type=3)
            # finite difference approximation, can also improve by
            # providing derivative directly
            if not gderiv:
                self._gderiv = [lambda u: (g(u + 0.001) - g(u))/0.001 for g in self._g]
            else:
                self._gderiv = gderiv
        elif boundaries == [-1, 1]:
            # TODO: figure out why DCT doesn't work
            self.fft = lambda s, **kwargs: dct(s, norm='ortho', type=1)
            self.ifft = lambda s, **kwargs: idct(s, norm='ortho', type=1)
            if not gderiv:
                self._gderiv = [lambda u: (g(u + 0.001) - g(u))/0.001]*len(self._g)
            else:
                self._gderiv = gderiv
        offset = 0.5 if boundaries == [1, -1] else 0
        self.derivative_matrix = ((np.arange(1, points+1) - offset)*np.pi)**2

    def homogenize(self, u, x, field):
        """
        Perform the transformation to a function with homogeneous boundary
        conditions.

        Parameters
        ----------
        u: np.ndarray
            Function value at the lattice points
        x: np.ndarray
            Lattice values
        """
        f, g = self._f[field], self._g[field]
        if self._boundaries == [1, -1]:
            return u - f - x*g(u[-1])
        elif self._boundaries == [-1, 1]:
            return u - x*f - (x - 1)*g(u[0])
        elif self._boundaries == [1, 1]:
            return u - (1 - x)*f - x*g

    def dehomogenize(self, v, x, u_boundary, field):
        """
        Transform back to a function with inhomogeneous boundaries

        Parameters
        ----------
        v: np.ndarray
            Function values at lattice points
        x: np.ndarray
            Lattice points
        u_boundary: float
            Value of inhomogeneous function at the Robin boundary, if there
            is one
        """
        f, g = self._f[field], self._g[field]
        if self._boundaries == [1, -1]:
            return v + f + x*g(u_boundary)
        elif self._boundaries == [-1, 1]:
            return v + x*f + (x - 1)*g(u_boundary)
        elif self._boundaries == [1, 1]:
            return v + (1 - x)*f + x*g

    def theta(self, x, boundary, boundary_deriv, field):
        """
        Computes inhomogeneous function theta, in the PDE: dv/dt = v'' + theta

        Parameters
        ----------
        x: np.ndarray
            Lattice points
        boundary: float
            Value of function u at boundary
        boundary_deriv: float
            Time derivative of function u at boundary
        """
        gderiv = self._gderiv[field]
        if self._boundaries == [1, -1]:
            return -x*gderiv(boundary)*boundary_deriv
        elif self._boundaries == [-1, 1]:
            return -(x - 1)*gderiv(boundary)*boundary_deriv
        else:
            return np.zeros(len(x))

    def boundary_iteration(self, x, v, dvdt, boundary, boundary_deriv, field):
        """
        Perform a single fixed-point iteration of the boundary
        values.

        Parameters
        ----------
        x: np.ndarray
            Lattice points
        v: np.ndarray
            Function value at lattice points
        dvdt: np.ndarray
            Derivative of v at lattice points
        boundary: float
            Previous boundary value of u
        boundary_deriv: float
            Previous derivative of boundary value
        """
        # Dirichlet - Robin
        f, g, gderiv = self._f[field], self._g[field], self._gderiv[field]
        if self._boundaries == [1, -1]:
            # Test: newton's algorithm for boundary
            p = v[-1] + f
            boundary = boundary \
                - (p + g(boundary) - boundary) / \
                (gderiv(boundary) - 1)
            #boundary = v[-1] + self._f + x[-1]*self._g(boundary)
            boundary_deriv = dvdt[-1] \
                + x[-1]*gderiv(boundary)*boundary_deriv
        # Robin - Dirichlet
        elif self._boundaries == [-1, 1]:
            boundary = v[0] + x[0]*f + (x[0] - 1)*g(boundary)
            boundary_deriv = dvdt[0] + \
                x[0]*f + (x[0] - 1)*gderiv(boundary)*boundary_deriv
        else:
            # If there are no Robin boundaries, zero these
            boundary, boundary_deriv = 0, 0
        return boundary, boundary_deriv