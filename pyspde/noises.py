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


class FourierMollifiedNoise:

    def __init__(self, mollifier, lattice, fields=1, seed=None):
        #self._eps = eps
        # TODO: generalize to L != 1
        self._mollifier = mollifier
        self._time = 0.0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        self._xs = lattice.points
        xs = np.linspace(self._xs[0]*2, self._xs[-1]*2, len(self._xs)*2)
        #u = np.exp(-(xs/(2*self._eps))**2)/(2*pi*self._eps)
        self._u = self._mollifier(xs)
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
        uhat = np.fft.rfft(self._u)
        sigma = sqrt(5*pi)/2 * np.abs(uhat)

        w = 0
        for _ in range(factor):
            fourier_noise = self._rng.normal(scale=sigma/sqrt(dt))
            w += np.fft.irfft(fourier_noise)

        self._value = w[:dimension]/factor
        self._time = t
        return self._value.reshape((1, dimension))

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)



class SpatioTemporalColoredNoise:

    def __init__(self, steps, dimension, space_mollifier, time_mollifier, seed=None):
        self._steps = steps
        self._dimension = dimension
        self._space_mollifier = space_mollifier
        self._time_mollifier = time_mollifier
        self._noise_field = np.zeros((steps, dimension))
        self._index = 0
        self._time = 0
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    def __call__(self, t, dx, dimension, average=False):
        if self._index == 0:
            self._index += 1
            self._time = t
            return self._noise_field[self._index-1]
        elif self._index == 1:
            dt = t - self._time
            self._noise_field = self._rng.normal(scale=1/sqrt(dt*dx), size=self._noise_field.shape)
            # mollify noise field in space
            #self._noise_field = np.convolve(self._noise_field*dx, self._space_mollifier, mode="same", axis=1)
            self._noise_field = np.apply_along_axis(lambda m: np.convolve(m, self._space_mollifier*dx, mode='same'), axis=1, arr=self._noise_field)
            # mollify noise field in time
            #self._noise_field = np.convolve(self._noise_field*dt, self._time_mollifier, mode="same", axis=0)
            self._noise_field = np.apply_along_axis(lambda m: np.convolve(m, self._time_mollifier*dt, mode='same'), axis=0, arr=self._noise_field)

            self._time = t
            self._index += 1
        return self._noise_field[self._index]

        



    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)

    