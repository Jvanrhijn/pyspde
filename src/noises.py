from abc import ABC, abstractmethod
from math import sqrt, exp, pi
import copy

import numpy as np
from scipy.stats import norm


class Noise(ABC):

    @abstractmethod
    def __call(self, t):
        raise NotImplementedError()


class WhiteNoise:

    def __init__(self, covariance, fields=1, seed=None):
        self._fields = fields
        if isinstance(covariance, np.ndarray):
            self._covariance = covariance
        else:
            self._covariance = np.eye(fields)*covariance
        #self._variance = variance
        self._time = 0.0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def __call__(self, t, dx, dimension, average=False):
        self._value = np.zeros((1, dimension))
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # special case: if called at the same time point, return same value as previous
        if dt == 0.0:
            return self._value

        factor = 2 if average else 1

        # generate white noise process
        if self._fields > 1:
            self._value = np.mean(self._rng.multivariate_normal(np.zeros(self._fields), self._covariance*factor/(dx*dt),
                                               size=(factor, dimension)), axis=0)
        else:
            # sampling from 1D normal distribution is likely faster,
            # so do that if there's only one field
            self._value = np.mean(self._rng.normal(scale=sqrt(factor*self._covariance[0]/(dx*dt)),
                                                size=(factor, dimension)), axis=0)

        self._time = t
        # TODO: generalize this to more fields
        return self._value.reshape((1, dimension))

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        if isinstance(cov, np.ndarray):
            self._covariance = cov 
        else:
            self._covariance = np.eye(self._fields)*cov

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)


class WhiteNoiseField:

    def __init__(self, covariance, dimension, seed=None):
        if isinstance(covariance, np.ndarray):
            self._covariance = covariance
        else:
            self._covariance = np.eye(dimension)*covariance
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

        factor = 2 if average else 1

        # generate white noise process
        self._value = np.mean(self._rng.multivariate_normal(np.zeros(self._dimension), 
            self._covariance*factor/(self._dx*dt),
            size=factor), axis=0)

        self._time = t
        return self._value.T

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov 

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)


class MollifiedWhiteNoise:

    def __init__(self, covariance, mollifier, fields=1, seed=None):
        self._fields = fields
        self._mollifier = mollifier
        if isinstance(covariance, np.ndarray):
            self._covariance = covariance
        else:
            self._covariance = np.eye(fields)*covariance
        #self._variance = variance
        # TODO: generalize to L != 1
        self._time = 0.0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def __call__(self, t, dx, dimension, average=False):
        self._value = np.zeros((1, dimension))
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # special case: if called at the same time point, return same value as previous
        if dt == 0.0:
            return self._value

        factor = 2 if average else 1

        # generate white noise process
        if self._fields > 1:
            self._value = np.mean(self._rng.multivariate_normal(np.zeros(self._fields), self._covariance*factor/(dx*dt),
                                               size=(factor, dimension)), axis=0)
        else:
            # sampling from 1D normal distribution is likely faster,
            # so do that if there's only one field
            self._value = np.mean(self._rng.normal(scale=sqrt(factor*self._covariance[0]/(dx*dt)),
                                                size=(factor, dimension)), axis=0)

        self._value = np.convolve(self._mollifier, self._value*dx, mode="same")
        self._time = t
        # TODO: generalize this to more fields
        return self._value.reshape((1, dimension))

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        if isinstance(cov, np.ndarray):
            self._covariance = cov 
        else:
            self._covariance = np.eye(self._fields)*cov

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)


class GaussianMollifiedNoise:

    def __init__(self, eps, fields=1, seed=None):
        self._eps = eps
        # TODO: generalize to L != 1
        self._time = 0.0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        self._value = None

    def __call__(self, t, dx, dimension, average=False):
        if self._value is None:
            self._value = np.zeros((1, dimension))
        if t < self._time:
            raise ValueError("Can't compute white noise into the past")
        dt = t - self._time
        # special case: if called at the same time point, return same value as previous
        if dt == 0.0:
            return self._value

        factor = 2 if average else 1

        dt /= factor

        n = dimension
        ns = np.arange(0, n+1)
        mus = np.arange(-n, n+1)
        coeff = np.sqrt(2*n*dx*np.exp(2*self._eps**2/dx**2 * (np.cos(2*pi*mus/(2*n)) - 1))) * sqrt(2*pi)

        w = 0
        for _ in range(factor):
            a = self._rng.normal(size=len(ns), scale=sqrt(0.5))
            b = self._rng.normal(size=len(ns), scale=sqrt(0.5))
            a[0] = 1
            b[0] = 0

            a_extend = np.concatenate((a[1:][::-1], a))
            b_extend = np.concatenate((-b[1:][::-1], b))
            rands = a_extend + 1j * b_extend

            noise = np.fft.irfft(np.fft.ifftshift(coeff * rands), norm='ortho') * sqrt(2*n)
            w += np.fft.fftshift(noise[:n]) * self._rng.normal(scale=sqrt(1/dt))

        w /= factor

        self._value = w / 16 * sqrt(2)
        self._time = t
        return self._value.reshape((1, dimension))

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

