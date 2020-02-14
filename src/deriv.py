import numpy as np

from src.spde import Lattice
from src.spde import Boundary


class DerivativeOperator:

    def __init__(self, order, lattice, boundaries):
        self._order = order
        self._dx = lattice.increment
        # TODO: implement different boundaries
        self._left = boundaries[0]
        self._right = boundaries[1]

    def __call__(self, u):
        if self._order == 1:
            derivative = np.zeros(u.shape)
            derivative[:, 1:-1] = (u[:, 2:] - u[:, :-2])/(2*self._dx)
            # handle boundaries: low end
            if self._left.kind() == Boundary.DIRICHLET:
                derivative[:, 0] = derivative[:, 1]
            else:
                raise NotImplementedError("Finite difference not implemented for given BC combination")

            # handle boundaries: high end
            if self._right.kind() == Boundary.DIRICHLET:
                derivative[:, -1] = derivative[:, -2]
            elif self._right.kind() == Boundary.ROBIN:
                derivative[:, -1] = self._right(u[:, -1])
            else:
                raise NotImplementedError("Finite difference not implemented for given BC combination")

        elif self._order == 2:
            raise NotImplementedError("Second-order FD not yet implemented")
            derivative = np.zeros(u.shape)
            derivative[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2])/self._dx**2
            derivative[:, 0] = derivative[:, 1]
            derivative[:, -1] = 2 * (self._right(u[:, -1])/self._dx -
                                  (u[:, -1] - u[:, -2])/self._dx**2)
        return derivative.reshape(u.shape)


class SpectralDerivative:

    def __init__(self, order, lattice, basis, boundaries):
        self._left = boundaries[0]
        self._right = boundaries[1]
        self._basis = basis
        self._lattice = lattice
        if order != 1:
            raise NotImplementedError("Only 1st order spectral derivative available")

    def __call__(self, u, xs):
        # get boundary value
        if self._left.kind() == Boundary.DIRICHLET and self._right.kind() == Boundary.DIRICHLET:
            boundary_deriv = self._right() - self._left()
            boundary = (self._left() + boundary_deriv*xs)
            boundary_lattice = self._left() + boundary_deriv*self._lattice.points
        else:
            boundary_deriv = 0
            boundary = self._left()
            boundary_lattice = boundary
        # convert to expansion
        vs = self._basis.coefficients_generic(u - boundary, xs)
        # compute derivatives of basis functions
        phi_deriv = self._basis(xs, derivative=True)
        # compute and return expansion derivative
        return (boundary_deriv + phi_deriv.T @ vs.T).reshape(u.shape)
