import numpy as np


class Integral:

    def __init__(self, function, lattice, axis):
        self._lattice = lattice
        self._axis = axis
        self._function = function

    def __call__(self, field):
        return np.trapz(field, self._lattice, axis=self._axis)

    def function(self):
        return self._function

    @property
    def lattice(self):
        return self._lattice

    @property
    def axis(self):
        return self._axis