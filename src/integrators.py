from abc import ABC, abstractmethod
import numpy as nump


class Integrator(ABC):

    def __init__(self):
        self._time = 0

    @abstractmethod
    def step(self, field, noise, linsolve, time_step, average=False):
        raise NotImplementedError()

    def ipsteps(self):
        return 1

class Midpoint(Integrator):

    def __init__(self, iterations=4):
        super().__init__()
        self._iterations = iterations

    def ipsteps(self):
        return 2

    def step(self, field, spde, linsolve, time_step, average=False):
        # propagate solution to t + dt/2
        a0 = linsolve.propagate_step(field)
        a = a0
        # perform midpoint iteration
        w = spde.noise(self._time + 0.5*time_step, average=average)
        for _ in range(self._iterations):
            a = a0 + 0.5*time_step * \
                spde.da(a, self._time + 0.5*time_step, w)
        # propagate solution to t + dt
        self._time += time_step
        return linsolve.propagate_step(2*a - a0)


class RK2(Integrator):

    def step(self, field, spde, linsolve, time_step, average=False):
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

    def step(self, field, spde, linsolve, time_step, average=False):
        field_bar = linsolve.propagate_step(field)
        w1 = spde.noise(self._time, average=average)
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