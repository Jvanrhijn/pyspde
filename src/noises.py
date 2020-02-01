from abc import ABC, abstractmethod
from math import sqrt
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
        self._dx = 1/dimension
        self._value = np.zeros(dimension)
        self._time = 0.0

    def __call__(self, t):
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # special case: if called at the same time point, return same value as previous
        if dt == 0.0:
            return self._value
        # generate white noise process
        self._value = norm.rvs(scale=sqrt(self._variance/(self._dx*dt)), size=self._dimension)
        self._time = t
        return self._value

