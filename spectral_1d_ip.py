from functools import partial
from math import sqrt, sin, cos

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits import mplot3d

import scipy.linalg
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.fftpack import dst, idst, dct, idct

font = {'family' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)


class Transform:
    """
    This class contains all logic required to perform an interaction picture
    step. It contains code to transform the function to homogeneous boundaries,
    to compute the inhomogeneous function $\theta$, and to compute
    one fixed-point iteration of the boundary values.
    """

    def __init__(self, boundaries, f, g, gderiv=None):
        """
        Parameters
        ----------
        boundaries: list
            List of boundary types. A '1' means Dirichlet, '-1' means Robin
        f: float 
            Fixed boundary value
        g: float or function
            Fixed boundary value if boundaries == [1, 1], or Robin
            boundary function otherwise
        """
        self._boundaries = boundaries
        self._f = f
        self._g = g
        if boundaries == [1, 1]:
            self.fft = lambda s, **kwargs: dst(s, norm='ortho', type=1)
            self.ifft = lambda s, **kwargs: idst(s, norm='ortho', type=1)
        elif boundaries == [1, -1]:
            self.fft = lambda s, **kwargs: dst(s, norm='ortho', type=3)
            self.ifft = lambda s, **kwargs: idst(s, norm='ortho', type=3)
            # finite difference approximation, can also improve by
            # providing derivative directly
            if not gderiv:
                self._gderiv = lambda u: (g(u + 0.001) - g(u))/0.001
            else:
                self._gderiv = gderiv
        elif boundaries == [-1, 1]:
            # TODO: figure out why DCT doesn't work
            self.fft = lambda s, **kwargs: dct(s, norm='ortho', type=1)
            self.ifft = lambda s, **kwargs: idct(s, norm='ortho', type=1)
            if not gderiv:
                self._gderiv = lambda u: (g(u + 0.001) - g(u))/0.001
            else:
                self._gderiv = gderiv
        offset = 0.5 if boundaries == [1, -1] else 0
        self.derivative_matrix = np.diag(
                ((np.arange(1, len(xs)+1) - offset)*np.pi)**2)

    def homogenize(self, u, x):
        """
        Perform the transformation to a function with homogeneous boundary
        conditions.

        Parameters
        ----------
        u: np.ndarray
            Function value at the lattice points
        x: np.ndarray
            Lattice values 
        """
        if self._boundaries == [1, -1]:
            return u - self._f - x*self._g(u[-1])
        elif self._boundaries == [-1, 1]:
            return u - x*self._f - (x - 1)*self._g(u[0])
        elif self._boundaries == [1, 1]:
            return u - (1 - x)*f - x*g

    def dehomogenize(self, v, x, u_boundary):
        """
        Transform back to a function with inhomogeneous boundaries
        
        Parameters
        ----------
        v: np.ndarray
            Function values at lattice points
        x: np.ndarray
            Lattice points
        u_boundary: float
            Value of inhomogeneous function at the Robin boundary, if there
            is one
        """
        if self._boundaries == [1, -1]:
            return v + self._f + x*self._g(u_boundary)
        elif self._boundaries == [-1, 1]:
            return v + x*self._f + (x - 1)*self._g(u_boundary)
        elif self._boundaries == [1, 1]:
            return v + (1 - x)*self._f + x*self._g

    def theta(self, x, boundary, boundary_deriv):
        """
        Computes inhomogeneous function theta, in the PDE: dv/dt = v'' + theta

        Parameters
        ----------
        x: np.ndarray
            Lattice points
        boundary: float
            Value of function u at boundary
        boundary_deriv: float
            Time derivative of function u at boundary
        """
        if self._boundaries == [1, -1]:
            return -x*self._gderiv(boundary)*boundary_deriv
        elif self._boundaries == [-1, 1]: 
            return -(x - 1)*self._gderiv(boundary)*boundary_deriv
        else:
            return np.zeros(len(x))

    def boundary_iteration(self, x, v, dvdt, boundary, boundary_deriv):
        """
        Perform a single fixed-point iteration of the boundary
        values.

        Parameters
        ----------
        x: np.ndarray
            Lattice points
        v: np.ndarray
            Function value at lattice points
        dvdt: np.ndarray
            Derivative of v at lattice points
        boundary: float
            Previous boundary value of u
        boundary_deriv: float
            Previous derivative of boundary value
        """
        # Dirichlet - Robin
        if self._boundaries == [1, -1]:
            # Test: newton's algorithm for boundary
            p = v[-1] + self._f
            boundary = boundary \
                - (p + self._g(boundary) - boundary)/(self._gderiv(boundary) - 1)
            #boundary = v[-1] + self._f + x[-1]*self._g(boundary)
            boundary_deriv = dvdt[-1] \
                    + x[-1]*self._gderiv(boundary)*boundary_deriv
        # Robin - Dirichlet
        elif self._boundaries == [-1, 1]:
            boundary = v[0] + x[0]*self._f + (x[0] - 1)*self._g(boundary)
            boundary_deriv = dvdt[0] + \
                    x[0]*self._f + (x[0] - 1)*self._gderiv(boundary)*boundary_deriv
        else:
            # If there are no Robin boundaries, zero these
            boundary, boundary_deriv = 0, 0
        return boundary, boundary_deriv



dim = 40
ns = np.arange(0, dim, 1)

dx = 1 / dim
#xs = np.arange(dx, 1+dx, dx)
xs = np.linspace(0, 1, dim)
ts = np.linspace(0, 5, 100)


# boundaries
alpha = -1
boundary_types = [1, -1]
g = lambda u: alpha*u
gderiv = lambda u: alpha
#g = 0
f = 1

# initial condition
u0 = lambda x: np.ones(len(x))*f


# Construct Transform object for this the given boundary conditions
transform = Transform(boundary_types, f, g, gderiv=gderiv)

# fixed-point iteration parameters
tolerance = 1e-10
max_iters = 100

# Solve directly using iterative interaction picture algorithm
solution = np.zeros((len(ts), len(xs)))
solution[0] = u0(xs)


"""
The code below solves the 1D diffusion equation with nonlinear boundaries
using a spectral approach, with the boundary conditions satisfied by
fixed-point iteration.
"""
D = transform.derivative_matrix
Dinv = np.linalg.inv(D)


for i, t in enumerate(ts[1:]):
    # Compute matrix propagator for this time-step
    v0 = transform.homogenize(solution[i], xs)
    v0hat = transform.fft(v0)
    propagator = expm(-D*(ts[i+1] - ts[i]))

    # Guess initial values for iterated parameters
    theta = np.zeros(len(xs))
    u_boundary = solution[i][-1]
    ut_boundary = 0

    for it1 in range(max_iters):
        theta_old = theta

        # Compute homogenized function in Fourier space using IP
        vhat = (np.eye(dim) - propagator) @ Dinv @ theta + propagator @ v0hat
        # Compute time-derivative in Fourier space using the ODE for vhat
        dvhat_dt = -D @ vhat + theta

        # Transform back to real space
        v = transform.ifft(vhat)
        dvdt = transform.ifft(dvhat_dt)

        u_boundary_old = u_boundary
        u_boundary, ut_boundary = transform.boundary_iteration(
                xs, v, dvdt, u_boundary, ut_boundary)

        # Compute new value for theta using new boundary values for u
        theta = transform.fft(transform.theta(xs, u_boundary, ut_boundary))
        
        # check convergence
        if (residue := abs(u_boundary - u_boundary_old)) < tolerance:
            break

        if it1 == max_iters-1:
            print("WARNING: outer loop max iters reached")

    # Compute solution value at this time point
    solution[i+1] = transform.dehomogenize(v, xs, u_boundary)



TS, XS = np.meshgrid(ts, xs)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(TS, XS, solution.T, rstride=1, cstride=1, cmap="viridis")

boundary_error = np.abs((solution[:, -2] - solution[:, -1])/(xs[-2] - xs[-1]) 
        - g(solution[:, -1]))
plt.figure()
plt.semilogy(ts, boundary_error)
plt.grid()
plt.xlabel("$t$")
plt.ylabel("$E(t)$")
#plt.savefig("/home/jesse/Dropbox/Uni/Stage/report/fig/heat_eq_boundary_error.eps")

#steady_state = lambda x: 1 - 0.171573*x
#plt.figure()
#plt.semilogy(xs, np.abs(steady_state(xs) - solution[-1, :]))
#plt.grid()
#plt.xlabel("$x$")
#plt.ylabel("$E^{ss}_n$")
#plt.savefig("/home/jesse/Dropbox/Uni/Stage/report/fig/heat_eq_ss_error.eps")


"""Analytical solution for g = const"""
"""
k = lambda n: (n - 0.5)*np.pi

def a(n):
    return 2*(np.sin(k(n)) - k(n)*np.cos(k(n)))/k(n)**2
    #return 2*np.sin(k(n))/k(n)


def u(x, t, nterms):
    ns = np.arange(1, nterms+1)
    return f + g(0)*x + sum(a(n)*np.exp(-k(n)**2*t) \
                * np.sin(k(n)*x) for n in ns)

num_time_slices = 5
indices = list(range(len(ts)))[::int(len(ts)/num_time_slices)]
exact = u(XS, TS, 500).T
plt.figure(figsize=(10, 5))
for i in indices:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(xs, solution[i, :], 'o', color=color)
    plt.plot(xs, exact[i, :], color=color, label=f"$t = {ts[i]:.2f}$")
plt.xlabel("$x$")
plt.ylabel("$u(x, t)$")
plt.grid()
plt.legend()
plt.savefig("/home/jesse/Dropbox/Uni/Stage/report/fig/heat_eq_compare.eps")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(TS, XS, exact.T, cmap="viridis")
"""

plt.show()
