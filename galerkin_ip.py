import sys
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
from scipy.fftpack import dst, idst, dstn, idstn
from scipy.optimize import root
from scipy.sparse.linalg import expm_multiply
import scipy.sparse as sparse

font = {'family' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

#matrix = sparse.csc_matrix
#spinv = sparse.linalg.inv
#expm = sparse.linalg.expm
#eye = sparse.eye
matrix = lambda x: x
spinv = np.linalg.inv
eye = np.eye

"""Idea: extend the grid to include the x = 0 function"""

# number of basis functions to use
dim = 10

ns = np.arange(0, dim, 1)

dx = 1 / dim
xs = np.linspace(dx, 1, dim)
ts = np.linspace(0, 5, 150)

# initial condition
u0 = lambda x: np.ones(len(x))

alpha = -5
# boundaries
g = lambda u: alpha*u**2
gderiv = lambda u: 2*alpha*u
gderiv = None
f = 1.

# fixed-point iteration parameters
tolerance = 1e-10
max_iters = 200

# Sine basis wave vector
def k(n):
    return (n - 0.5)*np.pi

"""Code for contructing Galerkin basis"""
def basis_hat(n, x):
    return np.piecewise(x,
            [
                np.logical_and((n)*dx < x, x <= (n+1)*dx),
                np.logical_and((n+1)*dx < x, x < (n+2)*dx),
            ],
            [
                lambda x: (x - ((n)*dx))/dx,
                lambda x: ((n+2)*dx - x)/dx,
                lambda x: 0
            ])

def basis_hat_deriv(n, x):
    return np.piecewise(x,
            [
                np.logical_and((n)*dx <= x, x <= (n+1)*dx),
                np.logical_and((n+1)*dx <= x, x <= (n+2)*dx),
            ],
            [
                lambda x: 1/dx,
                lambda x: -1/dx,
                lambda x: 0
            ])

def basis_parabola(n, x):
    return np.piecewise(x,
            [
                np.logical_and(n*dx <= x, x <= (n+2)*dx),
            ],
            [
                lambda x: 1/dx**2*(n*dx - x)*(x - (n+2)*dx),
                lambda x: 0,
            ])

def basis_parabola_deriv(n, x):
    return np.piecewise(x,
            [
                np.logical_and(n*dx <= x, x <= (n+2)*dx),
            ],
            [
                lambda x: 2/dx**2*((n+1)*dx - x),
                lambda x: 0,
            ])

def basis_sine(n, x):
    return np.sin(k(n)*x)

def basis_sine_deriv(n, x):
    return k(n)*np.cos(k(n)*x)

basis_spectral = [partial(basis_sine, n) for n in range(1, dim)]\
        + [lambda x: x]

basis_spectral_deriv = [partial(basis_sine_deriv, n) for n in range(1, dim)]\
        + [lambda x: np.ones(x.shape) if type(x) == np.ndarray else 1]

basis_hat = [partial(basis_hat, n) for n in range(dim)]
basis_hat_deriv = [partial(basis_hat_deriv, n) for n in range(dim)]


# Determine which basis to use (defaults to spectral)
if len(sys.argv) == 1 or sys.argv[1] == "spectral":
    basis = basis_spectral
    basis_deriv = basis_spectral_deriv
elif sys.argv[1] == "hat":
    basis = basis_hat
    basis_deriv = basis_hat_deriv
else:
    raise ValueError("Basis not supported")

"""End basis code"""


    
# construct matrices
a = np.zeros((dim, dim))
b = np.zeros((dim, dim))
phi = np.zeros((dim, dim))

phi_right = np.array([basis[i](xs[-1]) for i in range(dim)])


def G(u):
    return g(f + np.dot(u, phi_right))*phi_right


# build a manually
#np.fill_diagonal(a, 0.5)
#a[:-1, -1] = a[-1, :-1] = np.sin(k(ns[1:]))/k(ns[1:])**2
#a[-1, -1] = 1/3
#a = matrix(a)
#
#
### build b manually
#np.fill_diagonal(b, 0.5*k(ns[1:])**2)
#b[:-1, -1] = np.sin(k(ns[1:]))
#b[-1, :-1] = np.sin(k(ns[1:]))
#b[-1, -1] = 1
#b = matrix(b)


# matrices for converting between coefficients and solution
for i in range(dim):
    for j in range(dim):
        phi[i, j] = basis[j](xs[i])
        a[i, j] = quad(lambda x: basis[j](x)*basis[i](x), 0, 1)[0]
        b[i, j] = quad(lambda x: basis_deriv[j](x)*basis_deriv[i](x), 0, 1)[0]

sol_to_fem = np.linalg.inv(phi)
fem_to_sol = phi


# Solve directly using iterative interaction picture algorithm
b_inv = spinv(b)
D = spinv(a) @ b
solution = np.zeros((len(ts), len(xs)))

solution[0] = u0(xs)


def contract(v, v0, prop):
    return (eye(dim) - prop) @ b_inv @ G(v)\
            + prop @ v0

import tqdm
# assume uniform time step
dt = ts[1] - ts[0]
prop = expm(-D*dt)

S = (np.eye(dim) - prop) @ b_inv
q = S @ phi_right
qphi = np.outer(q, phi_right)

if gderiv is None:
    gderiv = lambda u: (g(u + 0.001) - g(u))/0.001

def jacobian(u):
    return gderiv(f + u @ phi_right) * np.outer(q, phi_right) - np.eye(dim)


def jacobian_inv(u):
    gprime = gderiv(f + u @ phi_right)
    return -(np.eye(dim) + gprime * qphi / (1 - gprime * phi_right @ q))


def newton_iterate(x, v0, it_max=100, tolerance=1e-10):
    func = lambda y: contract(y, v0, prop) - y
    #jac_inv = lambda y: np.linalg.inv(jacobian(y))
    jac_inv = jacobian_inv
    
    for it in range(it_max):
        x_old = x
        x = x - jac_inv(x) @ func(x)
        if np.linalg.norm(x - x_old) < tolerance:
            break
    return x


for i, t in tqdm.tqdm(enumerate(ts[1:]), total=len(ts)-1):
    # initial guess for v
    u = solution[i]
    v = sol_to_fem @ (u - f)
    v0 = v

    v = newton_iterate(v, v0)
    #for it in range(max_iters):
    #    v_old = v
    #    v = contract(v, v0, prop)
    #    residue = np.linalg.norm(v - v_old)/len(v)
    #    if residue < tolerance:
    #        break
    #    v_old = v
    #    if it == max_iters - 1:
    #        print("WARNING: max iterations reached")
    solution[i+1, :] = f + fem_to_sol @ v

TS, XS = np.meshgrid(ts, xs)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(TS, XS, solution.T, rstride=1, cstride=1, cmap="viridis")

boundary_error = np.abs((solution[:, -2] - solution[:, -1])/(xs[-2] - xs[-1]) 
        - g(solution[:, -1]))
plt.figure()
plt.semilogy(ts, boundary_error)
plt.grid()
plt.xlabel("t")
plt.ylabel("Boundary error")
#
#steady_state = lambda x: 1 - 0.171573*x
plt.figure()
#plt.semilogy(xs, np.abs(steady_state(xs) - solution[-1, :]))
plt.plot(xs, solution[-1, :])
plt.grid()
#plt.savefig("/home/jesse/Dropbox/Uni/Stage/report/fig/heat_eq_ss_error.eps")
plt.xlabel("$t$")
plt.ylabel("$E^{ss}_n$")

"""Analytical solution for f = 1, g = -1, u0(x) = 1"""
"""
def a(n):
    return 2*np.sin(k(n))/k(n)**2


def u(x, t, nterms):
    ns = np.arange(1, nterms+1)
    return f + g(0)*x + sum(a(n)*np.exp(-k(n)**2*t) \
                * np.sin(k(n)*x) for n in ns)


num_time_slices = 5
indices = list(range(len(ts)))[::int(len(ts)/num_time_slices)]
exact = u(XS, TS, 100).T
plt.figure(figsize=(10, 5))
for i in indices:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(xs, solution[i, :], 'o', color=color)
    plt.plot(xs, exact[i, :], color=color, label=f"$t = {ts[i]:.2f}$")
plt.xlabel("$x$")
plt.ylabel("$u(x, t)$")
plt.grid()
#plt.savefig("/home/jesse/Dropbox/Uni/Stage/report/fig/heat_eq_compare.eps")
plt.legend()

"""

plt.show()

""" These lines can be used to compute a solution using forward time integration
# Time derivative function
def du(u, t):
    cs = sol_to_fem @ (u - f)
    dcsdt = np.linalg.solve(a, G(cs) - b @ cs)
    return fem_to_sol @ dcsdt


u0 = lambda x: np.ones(len(x))
sol = odeint(du, u0(xs), ts)
"""
