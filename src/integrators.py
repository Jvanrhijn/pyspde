from abc import ABC, abstractmethod
import numpy as np

from src.spde import Boundary


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
        da = lambda a, t, w: problem.spde.drift(a) + problem.spde.volatility(a)*w
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
        da = lambda a, t, w: problem.spde.drift(a) + problem.spde.volatility(a)*w

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


class MidpointFEM(Integrator):

    def __init__(self, lattice, basis, problem, midpoint_iters=3):
        super().__init__()
        self._midpoint_iters = midpoint_iters
        self._m = basis.mass()
        self._n = basis.stiffness()*problem.spde.linear
        self._minv = np.linalg.inv(self._m)
        self._q = self._minv @ self._n
        points = self._m.shape[0]
        self._dt = None 
        self._dx = lattice.increment
        self._xs = lattice.points
        self._xs_midpoint = lattice.midpoints
        self._phi_midpoint = basis(self._xs_midpoint)
        self._phi_deriv_midpoint = basis(self._xs_midpoint, derivative=True)
        self._phi_right = basis(lattice.range[1] - np.finfo(float).eps)
        self._time = 0
        self._ident = np.eye(points)
        self._basis = basis
        
    def set_timestep(self, time_step):
        self._dt = time_step
        self._jacobian_inv = -np.linalg.inv(self._ident + self._dt*self._q)

    def step(self, field, problem, average=False):
        dx = problem.lattice.increment
        midpoints = len(problem.lattice.midpoints)
        w = problem.spde.noise(self._time + 0.5*self._dt, dx, midpoints, average=average)
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            boundary = problem.left() * np.ones(field.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            boundary = problem.left() + (problem.right() - problem.left())*(self._xs - problem.lattice.range[0])
        else:
            raise NotImplementedError("Boundary combination not yet implemented")

        # extract FEM coefficients
        vs = self._basis.coefficients(field - boundary)

        # for the first iteration, ignore the drift and volatility
        a = np.zeros(vs.shape)
        b = np.zeros(vs.shape)

        g = self.boundary(vs, problem) * problem.spde.linear

        # compute next timestep iteratively
        vbar = vs
        for _ in range(self._midpoint_iters):
            d = a + b
            # do a Newton iteration
            # TODO: compute the correct Jacobian, as it currently assumes
            # d is evaluated at the start of the time interval
            vbar -= (self._jacobian_inv @ (vs.T - self._dt * self._q @ (vbar.T  + d.T + g.T) \
                + d.T + g.T - vbar.T)).reshape(vbar.shape)
            # approximate the midpoint value
            vmidpoint = 0.5 * (vs + vbar)
            # compute the drift and volatility integrals at the midpoint
            a = self.drift_integral(vmidpoint, problem)
            b = self.volatility_integral(vmidpoint, w, problem)
        self._time += self._dt
        return (boundary.reshape(vs.shape) + self._basis.lattice_values(vbar)).reshape(field.shape)
            
    def drift_integral(self, vs, problem):
        u_midpoint = (problem.left() \
            + (self._phi_midpoint.T @ vs.T)).T
        return (self._minv.T @ (self._phi_midpoint * (problem.spde.drift(u_midpoint)*self._dx*self._dt)).sum(axis=1)) \
            .reshape(vs.shape)

    def volatility_integral(self, vs, w, problem):
        u_midpoint = (problem.left() \
            + (self._phi_midpoint.T @ vs.T)).T
        return (self._minv.T @ (self._phi_midpoint * (problem.spde.volatility(u_midpoint)*w*self._dx*self._dt)).sum(axis=1)) \
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


class ThetaScheme(Integrator):
    
    def __init__(self, theta, lattice, basis, problem):
        super().__init__()
        self._theta = theta
        self._m = basis.mass()
        self._n = basis.stiffness() * problem.spde.linear
        self._minv = np.linalg.inv(self._m)
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
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            boundary = problem.left() * np.ones(field.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            boundary = problem.left() + (problem.right() - problem.left())*(self._xs - problem.lattice.range[0])
        else:
            raise NotImplementedError("Boundary combination not yet implemented")
        vs = self._basis.coefficients(field - boundary)
        d = self.drift_integral(vs, problem) + self.volatility_integral(vs, w, problem)
        g = self.boundary(vs, problem)
        self._time += self._dt
        return (boundary.reshape(field.shape) \
            + self._basis.lattice_values((self._gamma @ (vs.T + d.T + g.T)).reshape(field.shape)))

    def drift_integral(self, vs, problem):
        u_midpoint = (problem.left() \
            + (self._phi_midpoint.T @ vs.T)).T
        return (self._minv.T @ (self._phi_midpoint * (problem.spde.drift(u_midpoint)*self._dx*self._dt)).sum(axis=1)) \
            .reshape(vs.shape)

    def volatility_integral(self, vs, w, problem):
        u_midpoint = (problem.left() \
            + (self._phi_midpoint.T @ vs.T)).T
        return (self._minv.T @ (self._phi_midpoint * (problem.spde.volatility(u_midpoint)*w*self._dx*self._dt)).sum(axis=1)) \
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