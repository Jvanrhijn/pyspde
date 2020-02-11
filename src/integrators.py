from abc import ABC, abstractmethod
import numpy as np


class Integrator(ABC):

    def __init__(self, time_step):
        self._time = 0
        self._time_step = time_step

    @abstractmethod
    def step(self, field, problem, average=False):
        raise NotImplementedError()

    def set_timestep(self, dt):
        self._time_step = dt

    def ipsteps(self):
        return 1

class Midpoint(Integrator):

    def __init__(self, linsolve, dt, iterations=4):
        super().__init__(dt)
        self._linsolve = linsolve
        self._linsolve.set_timestep(dt/2)
        self._iterations = iterations

    def set_timestep(self, dt):
        self._linsolve.set_timestep(dt/2)
        self._time_step = dt

    def ipsteps(self):
        return 2

    def step(self, field, problem, average=False):
        # propagate solution to t + dt/2
        time_step = self._time_step
        a0 = self._linsolve.propagate_step(field, problem)
        a = a0
        da = lambda a, t, w: problem.spde.drift(a) + problem.spde.volatility(a)*w
        # perform midpoint iteration
        w = problem.spde.noise(self._time + 0.5*time_step, average=average)
        for _ in range(self._iterations):
            a = a0 + 0.5*time_step * \
                da(a, self._time + 0.5*time_step, w)
        # propagate solution to t + dt
        self._time += time_step
        return self._linsolve.propagate_step(2*a - a0, problem)


class RK2(Integrator):

    def step(self, field, spde, linsolve, average=False):
        time_step = self._time_step
        field_bar = linsolve.propagate_step(field)
        w1 = spde.noise(self._time, average=average)
        d1 = time_step * linsolve.propagate_step(spde.da(field, self._time, w1))
        w2 = spde.noise(self._time + time_step, average=average) 
        d2 = time_step * spde.da(field_bar + d1, self._time + time_step, w2)
        self._time += time_step
        return field_bar + 0.5 * (d1 + d2)


class RK4(Integrator):

    def ipsteps(self):
        return 2

    def step(self, field, spde, linsolve, average=False):
        time_step = self._time_step
        field_bar = linsolve.propagate_step(field)
        w1 = spde.noise(self._time, average=average).T
        d1 = 0.5*time_step*linsolve.propagate_step(spde.da(field, self._time, w1))
        w2 = spde.noise(self._time + 0.5*time_step)
        d2 = 0.5*time_step*spde.da(field_bar + d1, self._time + 0.5*time_step, w2)
        d3 = 0.5*time_step*spde.da(field_bar + d2, self._time + 0.5*time_step, w2)
        w3 = spde.noise(self._time + time_step, average=average)
        d4 = 0.5*time_step*spde.da(
            linsolve.propagate_step(field_bar + 2 * d3),
            self._time + time_step,
            w3
        )
        self._time += time_step
        return linsolve.propagate_step(
            field_bar + (d1 + 2*(d2 + d3))/3 + d4/3
        )

    def set_timestep(self, dt):
        self._linsolve.set_timestep(dt/2)
        self._time_step = dt



class ThetaScheme(Integrator):
    
    def __init__(self, theta, lattice, basis, timestep):
        self._theta = theta
        self._m = basis.mass()
        self._n = basis.stiffness()
        self._minv = np.linalg.inv(self._m)
        self._q = self._minv @ self._n
        points = self._m.shape[0]
        self._dt = timestep
        self._dx = 1/points
        self._xs_midpoint = lattice.midpoints
        self._phi_midpoint = basis(self._xs_midpoint)
        self._phi_deriv_midpoint = basis(self._xs_midpoint, derivative=True)
        self._phi_right = basis(lattice.range[1])
        self._time = 0
        self._ident = np.eye(points)
        self._gamma = self._ident - self._dt*self._q \
               @ np.linalg.inv(self._ident + theta * self._dt * self._q)
        self._basis = basis
        
    def set_timestep(self, time_step):
        self._dt = time_step
        self._gamma = self._ident - self._dt*self._q\
            @ np.linalg.inv(self._ident + self._theta * self._dt * self._q)
        
    """
    NOTE: most likely the noise is somehow being treated incorrectly,
          leading to large sampling errors which are being interpreted as step errors
    """
    def step(self, field, problem, average=False):
        w = problem.spde.noise(self._time + 0.5*self._dt, average=average)
        #return self._step_midpoint(field, w)
        return self._step(field, problem, w)
            
    def _step(self, field, problem, w):
        #vs = sol_to_fem @ (field - (left + (right - left)*xs)).T
        left = problem.left()
        # TODO generalize to more general boundary conditions
        vs = self._basis.coefficients(field - left)
        d = self.integrals(vs, w, problem)
        g = self.boundary(vs, problem)
        self._time += self._dt
        return (left + self._basis.lattice_values(self._gamma @ (vs.T + d.T + g.T))).reshape(field.shape)
    
    def integrals(self, vs, w, problem):
        u_midpoint = problem.left() \
            + (self._phi_midpoint.T @ vs.T)
        return (self._minv @ (((problem.spde.drift(u_midpoint)*self._dx*self._dt \
                             + problem.spde.volatility(u_midpoint)*w*self._dx*self._dt) \
                             * self._phi_midpoint)).sum(axis=1)).reshape(vs.shape)
    
    def boundary(self, vs, problem):
        #return (left - right) * self._dx * self._dt * (self._minv @ self._phi_deriv_midpoint).sum(axis=1)\
        #    .reshape(vs.shape)
        left = problem.left()
        g = problem.right
        coefficients = vs.reshape(self._phi_right.shape)
        return (self._minv @ (g(left + self._phi_right @ coefficients)*self._phi_right*self._dt)).reshape(vs.shape)