from abc import ABC, abstractmethod
from functools import partial
from math import sin, cos, sqrt
import copy

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import norm


class WhiteNoise:

    def __init__(self, variance, dimension):
        self._variance = variance
        self._dimension = dimension
        self._value = 0.0
        self._time = 0.0

    def __call__(self, t):
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        self._value += norm.rvs(scale=self._variance*dt, size=self._dimension)
        self._time = t
        return self._value


class SPDE:

    def __init__(self, linear, da, noise, points, left, right):
        self.noise = noise
        self.linear = linear
        self.da = da
        self.left = left
        self.right = right
        self.points = points


class LinearSolver:

    @abstractmethod
    def propagate_step(self, u):
        pass


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
        # matrices for converting between coefficients and solution
        for i in range(dim):
            for j in range(dim):
                phi[i, j] = self._basis_spectral[j](self._xs[i])
                self._a[i, j] = quad(lambda x: self._basis_spectral[j](x)*self._basis_spectral[i](x), 0, 1)[0]
                self._b[i, j] = quad(lambda x: self._basis_spectral_deriv[j](x)*self._basis_spectral_deriv[i](x), 0, 1)[0]
        self._fem_to_sol = phi
        self._sol_to_fem = np.linalg.inv(phi)

        # more precomputation of useful quantities
        self._d = np.linalg.inv(self._a) @ self._b
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


class TrajectorySolver:

    def __init__(self, spde, steps, tmax, initial, linear_solver):
        self._spde = spde
        self._steps = steps
        self._dt = tmax/steps
        self._dx = 1/spde.points
        self._xs = np.linspace(self._dx, 1, spde.points)
        self._initial = initial
        # linear solver should propagate half time-step for MP algorithm
        self._linear_solver = linear_solver(spde, self._dt/2)
        self._solution = np.zeros((steps+1, spde.points))
        self._solution[0] = initial
        self._time = 0

    def solve(self):
        noise = self._spde.noise
        midpoints_iters = 3
        for i in range(self._steps):
            # propagate solution to t + dt/2
            a0 = self._linear_solver.propagate_step(self._solution[i])
            a = a0 
            # perform midpoint iteration
            for it in range(midpoints_iters):
                a = a0 + 0.5*self._dt * self._spde.da(a, self._time + 0.5*self._dt, noise)
            # propagate solution to t + dt
            self._solution[i+1] = self._linear_solver.propagate_step(2*a - a0)
            self._time += self._dt

    @property
    def solution(self):
        return self._solution


class EnsembleSolver:

    def __init__(self, trajectory_solver, ensembles):
        self._trajectory_solvers = [copy.deepcopy(trajectory_solver) for _ in range(ensembles)]
        self._ensembles = ensembles
        self._storage = np.zeros((self._ensembles, *trajectory_solver.solution.shape))
        self.mean = None
        self.square = None

    def solve(self):
        for ensemble in range(self._ensembles):
            self._trajectory_solvers[ensemble].solve()
            trajectory = self._trajectory_solvers[ensemble].solution
            self._storage[ensemble] = trajectory
        self.mean = np.mean(self._storage, axis=0)
        self.square = np.mean(self._storage**2, axis=0)


class Visualizer:

    def __init__(self, solution, trange, srange):
        self._solution = solution
        self._trange = trange
        self._srange = srange
        steps, points = solution.shape
        dx = 1/points
        self._xs = np.linspace(self._srange[0] + dx, self._srange[1], points)
        self._ts = np.linspace(*self._trange, steps)

    def surface(self):
        xs, ts = np.meshgrid(self._xs, self._ts)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(xs, ts, self._solution, cmap="viridis")
        return fig, ax

    def steady_state(self):
        ss = self._solution[-1]
        fig, ax = plt.subplots(1)
        ax.plot(self._xs, ss)
        ax.grid(True)
        return fig, ax


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
        matrix[:-1, :-1] = np.diag(np.ones(dim-2), k=1) - np.diag(np.ones(dim-2), k=-1)
        matrix[0, :] = matrix[1, :]
        matrix[-1, -1] = 2*self._dx * self._right(u[-1]) / u[-1]
        matrix[-2, -1] = 1
        du = 1/(2*self._dx) * matrix @ u
        return du


if __name__ == "__main__":
    coeff = 1
    points = 20
    steps = 100
    tmax = 5

    sigma = sqrt(2)
    k = -1

    f = 1
    #g = lambda u: (k - sigma**2)*u
    g = lambda u: -0.5*sigma*u*np.exp(-u**2)

    u0 = np.ones(points)
    noise = WhiteNoise(1, points)

    d1 = DerivativeOperator(1, 1/points, f, g)

    da = lambda a, t, w: -d1(a)**2/a + a*sigma*sqrt(2)*w(t)
    da_linmult = lambda a, t, w: -d1(a)**2 * a/(1 + a**2) - (k - sigma**2)**2 * a / (1 + a**2) + sigma * np.sqrt(2*(1 + a**2))*w(t)
    da_gaussian = lambda a, t, w: a*d1(a)**2 + sigma**4 / 4 * np.exp(-2*a**2) * (a**3 - 1.5*a) + sigma * np.exp(-a**2/2)*w(t)*sqrt(2)

    spde = SPDE(coeff, da_gaussian, noise, points, f, g)

    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver)
    ensemble_solver = EnsembleSolver(solver, 10)
    ensemble_solver.solve()
    mean = ensemble_solver.mean
    square = ensemble_solver.square

    vis = Visualizer(mean, (0, tmax), (0, 1))
    vis2 = Visualizer(square, (0, tmax), (0, 1))

    fig, ax = vis.surface()
    fig2, ax2 = vis.steady_state()
    fig3, ax3 = vis2.steady_state()
    #xs = np.linspace(1/points, 1, points)
    #ax2.plot(xs, np.exp(k*xs), 'o')
    #ax3.plot(xs, np.exp(2*k*xs), 'o')

    plt.show()
