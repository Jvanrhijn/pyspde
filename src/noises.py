from abc import ABC, abstractmethod
import copy
import numpy as np
from scipy.stats import norm


class Noise(ABC):

    @abstractmethod
    def __call(self, t):
        raise NotImplementedError()


class WhiteNoise:

    def __init__(self, variance, dimension):
        self._variance = variance
        self._dimension = dimension
        self._value = np.zeros(dimension)
        self._time = 0.0

    def __call__(self, t, xs):
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # generate spatially delta-correlated white noise process
        value_new = copy.deepcopy(self._value)
        for j, x in enumerate(xs[1:]):
            dx = x - xs[j]
            value_new[j+1] = value_new[j] + norm.rvs(scale=self._variance*dx)\
                * norm.rvs(scale=self._variance*dt)
        self._value += value_new
        self._time = t
        return self._value

