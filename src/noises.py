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
        # TODO: generalize to L != 1
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

        factor = 2 if average else 1

        # generate white noise process
        # TODO: investigate wether the issues might be due to
        # the spatial noise not being brownian motion in truth
        
        # generate spatially correlated noise
        value = np.zeros((2, self._dimension))

        noise = self._rng.normal(scale=sqrt(factor*self._variance/(self._dx*dt)), size=(factor, self._dimension-1))
        for j in range(1, self._dimension):
            value[:, j] = value[:, j-1] + self._dx * noise[:, j-1]
        
        self._value = np.mean(value, axis=0)

        #self._value = np.mean(self._rng.normal(scale=sqrt(factor*self._variance/(self._dx*dt)),
        #                                       size=(factor, self._dimension)), axis=0)

        self._time = t
        return self._value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)