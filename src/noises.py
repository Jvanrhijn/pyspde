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

    def __init__(self, variance, dimension, seed=None):
        self._variance = variance
        self._dimension = dimension
        self._dx = 1/dimension
        self._value = np.zeros(dimension)
        self._time = 0.0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def __call__(self, t, average=False):
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # special case: if called at the same time point, return same value as previous
        if dt == 0.0:
            return self._value
        # generate white noise process
        factor = 2 if average else 1
        self._value = np.mean(self._rng.normal(scale=sqrt(factor*self._variance/(self._dx*dt)),
                                               size=(factor, self._dimension)), axis=0)
        self._time = t
        return self._value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)