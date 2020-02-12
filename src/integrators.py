from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt

from src.basis import FiniteElementBasis
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

    def __init__(self, lattice, basis, problem, solver=opt.broyden1):
        super().__init__()
        self._m = basis.mass()
        self._n = basis.stiffness() * problem.spde.linear
        self._minv = np.linalg.inv(self._m)
        self._q = self._minv @ self._n
        self._dt = None 
        self._dx = lattice.increment
        self._xs = lattice.points
        self._xs_midpoint = lattice.midpoints
        self._phi_midpoint = basis(self._xs_midpoint)
        self._phi_deriv_midpoint = basis(self._xs_midpoint, derivative=True)
        self._phi_right = basis(lattice.range[1] - np.finfo(float).eps)
        self._time = 0
        self._solver = solver
        self._basis = basis
        
    def set_timestep(self, time_step):
        self._dt = time_step

    def step(self, field, problem, average=False):
        dx = problem.lattice.increment
        midpoints = len(problem.lattice.midpoints)
        w = problem.spde.noise(self._time + 0.5*self._dt, dx, midpoints, average=average)

        boundary = self.get_boundary(field, problem)

        # extract FEM coefficients
        vs = self._basis.coefficients(field - boundary).flatten()

        # function whose root to find for the update function
        def minfunc(x):
            x = x.flatten()
            vmid = 0.5 * (vs + x)
            a = self.drift_integral(vmid, problem)
            b = self.volatility_integral(vmid, w, problem)
            d = a + b
            g = self.boundary(vmid, problem) * problem.spde.linear
            return (vs.T - self._dt * self._q @ (vmid.T + d.T + g.T) + d.T + g.T - x.T).reshape(x.shape)
            
        vbar = self._solver(minfunc, vs)

        self._time += self._dt
        return (boundary.reshape(vs.shape) + self._basis.lattice_values(vbar)).reshape(field.shape)

    def drift_integral(self, vs, problem):
        if problem.spde.linear != 0:
            boundary = self.get_boundary(vs, problem)
            u_midpoint = (boundary \
                + (self._phi_midpoint.T @ vs.T)).T
            return (self._minv.T @ (self._phi_midpoint * (problem.spde.drift(u_midpoint)*self._dx*self._dt)).sum(axis=1)) \
                .reshape(vs.shape)
        else:
            return problem.spde.drift(vs)*self._dt

    def volatility_integral(self, vs, w, problem):
        boundary = self.get_boundary(vs, problem)
        u_midpoint = (boundary \
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

    def get_boundary(self, field, problem):
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return problem.left() * np.ones(field.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            return problem.left() + (problem.right() - problem.left())*(self._xs - problem.lattice.range[0])
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
        boundary = self.get_boundary(field, problem)
        vs = self._basis.coefficients(field - boundary)
        d = self.drift_integral(vs, problem) + self.volatility_integral(vs, w, problem)
        g = self.boundary(vs, problem) * problem.spde.linear
        self._time += self._dt
        return (boundary.reshape(field.shape) \
            + self._basis.lattice_values((self._gamma @ (vs.T + d.T + g.T)).reshape(field.shape)))

    def drift_integral(self, vs, problem):
        if problem.spde.linear != 0:
            boundary = self.get_boundary(vs, problem)
            u_midpoint = (boundary \
                + (self._phi_midpoint.T @ vs.T)).T
            return (self._minv.T @ (self._phi_midpoint * (problem.spde.drift(u_midpoint)*self._dx*self._dt)).sum(axis=1)) \
                .reshape(vs.shape)
        else:
            return problem.spde.drift(vs)*self._dt

    def volatility_integral(self, vs, w, problem):
        boundary = self.get_boundary(vs, problem)
        u_midpoint = (boundary \
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

    def get_boundary(self, field, problem):
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return problem.left() * np.ones(field.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            return problem.left() + (problem.right() - problem.left())*(self._xs - problem.lattice.range[0])
        else:
            raise NotImplementedError("Boundary combination not yet implemented")


class DifferentialWeakMethod(Integrator):

    def __init__(self, lattice, problem, method="midpoint"):
        super().__init__()
        print("Warning: DifferentialWeakMethod is experimental!")
        self._basis = FiniteElementBasis(lattice, [problem.left, problem.right])
        self._m = self._basis.mass()
        self._minv = np.linalg.inv(self._m)
        self._n = self._basis.stiffness()
        self._xs_midpoint = lattice.midpoints
        self._phi_midpoint = self._basis(self._xs_midpoint)
        self._phi_right = self._basis(lattice.range[1]-np.finfo(float).eps)
        self._dx = lattice.increment
        self._dt = 0

    def set_timestep(self, dt):
        self._dt = dt

    def dv(self, vs, problem, average=False):
        cov = self.covariance(vs, problem)*problem.spde.noise.covariance
        factor = 2 if average else 1
        w = np.mean(problem.spde.noise._rng.multivariate_normal(
            np.zeros(vs.size), factor*cov/(self._dx*self._dx), size=(factor, 1)),
             axis=0)
        a = self.drift(vs, problem).reshape(vs.shape)
        return (self._minv @ (a.T + w.T)).reshape(vs.shape)

    def covariance(self, vs, problem):
        b = problem.spde.volatility
        # TODO: arbitrary boundaries
        u_midpoint = (problem.left() + self._phi_midpoint.T @ vs.T).reshape(vs.shape)
        off_diag = b(u_midpoint[:, 1:])**2
        diag = np.zeros(u_midpoint.size)
        diag[:-1] = b(u_midpoint[:, 1:])**2 + b(u_midpoint[:, :-1])**2
        diag[-1] = b(u_midpoint[:, -1])**2
        cov = 0.25/self._dx * (
            np.diag(off_diag.flatten(), k=1) + np.diag(off_diag.flatten(), k=-1) + np.diag(diag.flatten()))
        return cov

    def drift(self, vs, problem):
        u_midpoint = (problem.left() + self._phi_midpoint.T @ vs.T).reshape(vs.shape)
        a = problem.spde.drift(u_midpoint)
        u_right = problem.left() + vs @ self._phi_right.T
        return (self._phi_midpoint * a).sum(axis=1) * self._dx \
            + problem.right(u_right)*self._phi_right \
            - (self._n @ vs.T).reshape(vs.shape)

    def step(self, field, problem, average=False):
        vs = self._basis.coefficients(field - problem.left())
        time_step = self._dt
        v0 = vs
        # perform midpoint iteration
        for _ in range(3):
            vs = v0 + 0.5*time_step * \
                self.dv(vs, problem, average=average)
        self._time += time_step
        return (problem.left() + self._basis.lattice_values(2*vs - v0)).reshape(field.shape)

    def get_boundary(self, field, problem):
        if problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.ROBIN:
            return problem.left() * np.ones(field.shape)
        elif problem.left.kind() == Boundary.DIRICHLET and problem.right.kind() == Boundary.DIRICHLET:
            return problem.left() + (problem.right() - problem.left())*(self._xs - problem.lattice.range[0])
        else:
            raise NotImplementedError("Boundary combination not yet implemented")
