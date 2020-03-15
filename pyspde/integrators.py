from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt

from pyspde.basis import FiniteElementBasis
from pyspde.boundary import Boundary


class Integrator(ABC):

    def __init__(self):
        self._time = 0

    @abstractmethod
    def step(self, field, problem, average=False):
        raise NotImplementedError()

    @abstractmethod
    def set_timestep(self, dt):
        raise NotImplementedError()


class MidpointIP(Integrator):

    def __init__(self, linsolve, iterations=4):
        super().__init__()
        self._linsolve = linsolve
        self._iterations = iterations
        self._time_step = 0

    def set_timestep(self, dt):
        self._linsolve.set_timestep(dt/2)
        self._time_step = dt

    def step(self, field, problem, average=False):
        dx = problem.lattice.increment
        dimension = len(problem.lattice.points)
        # propagate solution to t + dt/2
        time_step = self._time_step
        a0 = self._linsolve.propagate_step(field, problem)
        a = a0
        da = lambda a, t, w: problem.spde.drift(a, self._time) + problem.spde.volatility(a, self._time)*w
        # perform midpoint iteration
        w = problem.spde.noise(self._time + 0.5*time_step, dx, dimension, average=average)
        for _ in range(self._iterations):
            a = a0 + 0.5*time_step * \
                da(a, self._time + 0.5*time_step, w).reshape(a0.shape)
        # propagate solution to t + dt
        self._time += time_step
        return self._linsolve.propagate_step(2*a - a0, problem)


class RK4IP(Integrator):

    def __init__(self, linsolve):
        super().__init__()
        self._linsolve = linsolve

    def step(self, field, problem, average=False):
        dx = problem.lattice.increment
        dimension = len(problem.lattice.points)
        da = lambda a, t, w: problem.spde.drift(a, t) + problem.spde.volatility(a, t)*w

        time_step = self._time_step
        field_bar = self._linsolve.propagate_step(field, problem).reshape(field.shape)
        w1 = problem.spde.noise(self._time, dx, dimension, average=average).reshape(field.shape)
        d1 = 0.5*time_step*self._linsolve.propagate_step(da(field, self._time, w1), problem).reshape(field.shape)

        w2 = problem.spde.noise(self._time + 0.5*time_step, dx, dimension, average=average).reshape(field.shape)
        d2 = 0.5*time_step*da(field_bar + d1, self._time + 0.5*time_step, w2).reshape(field.shape)

        d3 = 0.5*time_step*da(field_bar + d2, self._time + 0.5*time_step, w2)
        w3 = problem.spde.noise(self._time + time_step, dx, dimension, average=average).reshape(field.shape)

        d4 = 0.5*time_step*da(
            self._linsolve.propagate_step(field_bar + 2 * d3, problem),
            self._time + time_step,
            w3
        ).reshape(field.shape)
        self._time += time_step

        return self._linsolve.propagate_step(
            field_bar + (d1 + 2*(d2 + d3))/3 + d4/3,
            problem
        ).reshape(field.shape)

    def set_timestep(self, dt):
        self._linsolve.set_timestep(dt/2)
        self._time_step = dt


class ThetaScheme(Integrator):
    
    def __init__(self, theta, lattice, basis, problem):
        super().__init__()
        self._theta = theta
        self._m = basis.mass()
        self._n = basis.stiffness() * problem.spde.linear
        self._minv = np.linalg.inv(self._m)

        # TODO: generalize for different boundary conditions
        points = len(lattice.points)
        dx = lattice.increment
        self._q = self._minv @ self._n
        points = self._m.shape[0]
        self._dt = 0
        self._dx = lattice.increment
        self._xs = lattice.points
        self._xs_midpoint = lattice.midpoints
        self._phi_midpoint = basis(self._xs_midpoint)
        self._phi_deriv_midpoint = basis(self._xs_midpoint, derivative=True)
        self._phi_right = basis(lattice.range[1] - np.finfo(float).eps)
        self._time = 0
        self._ident = np.eye(points)
        self._basis = basis
        self._psis = [basis.member(i) for i in range(-1, points-1)]
        self._psi_midpoint = np.array([psi(self._xs_midpoint) for psi in self._psis])
        
    def set_timestep(self, time_step):
        self._dt = time_step
        self._gamma = self._ident - self._dt*self._q\
            @ np.linalg.inv(self._ident + self._theta * self._dt * self._q)
        
    def step(self, field, problem, average=False):
        dx = problem.lattice.increment
        midpoints = len(problem.lattice.midpoints)
        # compute noise at the start of the interval
        w = problem.spde.noise(self._time, dx, midpoints, average=average)
        return self._step(field, problem, w)
            
    def _step(self, field, problem, w):
        boundary = self.get_boundary(field, self._xs, problem)[0]
        vs = self._basis.coefficients(field - boundary)
        d = self.drift_integral(vs, problem) + self.volatility_integral(vs, w, problem)
        g = self.boundary(vs, problem) * problem.spde.linear
        self._time += self._dt
        return (boundary.reshape(field.shape) \
            + self._basis.lattice_values((self._gamma @ (vs + d + g).T).reshape(field.shape)))

    def drift_integral(self, vs, problem):
        if problem.spde.linear != 0:
            boundary = self.get_boundary(vs, self._xs_midpoint, problem)[0]
            u_midpoint = (boundary \
                + (self._phi_midpoint.T @ vs.T).T)
            return (self._minv.T @ (self._phi_midpoint * (problem.spde.drift(u_midpoint, self._time)*self._dx*self._dt)).sum(axis=1)) \
                .reshape(vs.shape)
        else:
            return problem.spde.drift(vs)*self._dt

    def volatility_integral(self, vs, w, problem):
        boundary = self.get_boundary(vs, self._xs_midpoint, problem)[0]
        u_midpoint = (boundary \
            + (self._phi_midpoint.T @ vs.T).T)
        return (self._minv.T @ (self._phi_midpoint * (problem.spde.volatility(u_midpoint, self._time)*w*self._dx*self._dt)).sum(axis=1)) \
            .reshape(vs.shape)


    def boundary(self, vs, problem):
        coefficients = vs.reshape(self._phi_right.shape)
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return (self._minv @ (problem.right(problem.left() + self._phi_right @ coefficients)\
                *self._phi_right*self._dt))\
                .reshape(vs.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            return (problem.left() - problem.right()) * self._dx * self._dt \
                * (self._minv @ self._phi_deriv_midpoint).sum(axis=1)\
                .reshape(vs.shape)
        else:
            raise NotImplementedError("Boundary combination not yet implemented")

    def get_boundary(self, field, xs, problem):
        delta = problem.lattice.range[1] - problem.lattice.range[0]
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return problem.left() * np.ones(field.shape), 0
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            slope = (problem.right() - problem.left())/delta
            return problem.left() + slope*(xs - problem.lattice.range[0]), slope
        else:
            raise NotImplementedError("Boundary combination not yet implemented")