from abc import ABC, abstractmethod
from functools import partial
from math import sin, cos, sqrt
import copy
from multiprocessing import Process, Queue
from datetime import datetime

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import norm

import cupy as cp

try:
    import tqdm
except ImportError:
    tqdm = lambda s, *args, **kwargs: s


class SPDE:

    def __init__(self, linear, da, noise, points, left, right, right_deriv=None):
        self.noise = noise
        self.linear = linear
        self.da = da
        self.left = left
        self.right = right
        if right_deriv is not None:
            self.right_deriv = right_deriv
        else:
            self.right_deriv = lambda u: (right(u + 0.001) - right(u))/0.001
        self.points = points


class TrajectorySolver:

    def __init__(self, spde, steps, tmax, initial, linear_solver):
        self._spde = spde
        self._steps = steps
        self._dt = tmax/steps
        self._dx = 1/spde.points
        self._xs = np.linspace(self._dx, 1, spde.points)
        self._initial = initial
        # linear solver should propagate half time-step for MP algorithm
        self._linear_solver = linear_solver(spde, self._dt/2)
        self._solution = np.zeros((steps+1, spde.points))
        self._solution[0] = initial
        self._time = 0

    def solve(self):
        noise = self._spde.noise
        midpoints_iters = 4
        for i in range(self._steps):
            # propagate solution to t + dt/2
            if self._spde.linear != 0:
                a0 = self._linear_solver.propagate_step(self._solution[i])
            else:
                a0 = self._solution[i]
            a = a0 
            # perform midpoint iteration
            w = noise(self._time + 0.5*self._dt)
            for _ in range(midpoints_iters):
                a = a0 + 0.5*self._dt * self._spde.da(a, self._time + 0.5*self._dt, w)
            # propagate solution to t + dt
            if self._spde.linear != 0.0:
                self._solution[i+1] = self._linear_solver.propagate_step(2*a - a0)
            else:
                self._solution[i+1] = 2*a - a0
            self._time += self._dt

    @property
    def solution(self):
        return self._solution


class EnsembleSolver:

    def __init__(self, trajectory_solver, ensembles, processes=1):
        self._trajectory_solvers = [copy.deepcopy(trajectory_solver) for _ in range(ensembles)]
        self._ensembles = ensembles
        self._storage = np.zeros((self._ensembles, *trajectory_solver.solution.shape))
        self.mean = None
        self.square = None
        self.sample_error = None
        self.square_sample_error = None
        self._processes = processes

    def solve(self):
        queue = Queue() 
        # split the solver list into chunks
        solver_chunks = list(self.chunks(self._trajectory_solvers, self._processes))
        for thread, chunk in enumerate(solver_chunks):
            p = Process(target=self.solve_trajectory, args=(queue, thread, chunk))
            p.start()
        start_index = 0
        for chunk in solver_chunks:
            # obtain individual trajectory results from queue
            results = queue.get()
            num_results = len(results)
            # calculate range in internal trajectory storage
            r = (start_index, start_index + num_results)
            start_index = start_index + num_results
            # store results
            self._storage[r[0]:r[1]] = results
    
        # calculate some moments and errors
        self.mean = np.mean(self._storage, axis=0)
        self.square = np.mean(self._storage**2, axis=0)
        self.sample_error = np.std(self._storage, axis=0)/sqrt(self._ensembles - 1)
        self.square_sample_error = np.std(self._storage**2, axis=0)/sqrt(self._ensembles - 1)

    @staticmethod
    def solve_trajectory(queue, threadnr, solvers):
        results = []
        start_time = datetime.now()
        for snr, solver in enumerate(solvers):
            print(f"Thread {threadnr+1}: solving trajectory {snr+1}/{len(solvers)}, time = {datetime.now() - start_time}")
            solver.solve()
            results.append(solver.solution)
        queue.put(results)

    @staticmethod
    def chunks(l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]


class Visualizer:

    def __init__(self, solution, trange, srange, error=None):
        self._solution = solution
        self._error = error
        self._trange = trange
        self._srange = srange
        steps, points = solution.shape
        dx = 1/points
        self._xs = np.linspace(self._srange[0] + dx, self._srange[1], points)
        self._ts = np.linspace(*self._trange, steps)

    def surface(self, *args, **kwargs):
        xs, ts = np.meshgrid(self._xs, self._ts)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(xs, ts, self._solution, *args, cmap="viridis", **kwargs)
        return fig, ax

    def steady_state(self, *args, **kwargs):
        ss = self._solution[-1]
        fig, ax = plt.subplots(1)
        if self._error is not None:
            error = self._error[-1]
            ax.fill_between(self._xs, ss-error[-1], ss+error[-1], alpha=0.25)
        ax.plot(self._xs, ss, *args, **kwargs)
        ax.grid(True)
        return fig, ax

    @property
    def xaxis(self):
        return self._xs


class DerivativeOperator:

    def __init__(self, order, dx, left, right):
        self._order = order
        self._dx = dx
        self._left = left
        self._right = right
        du = 0.001
        self._right_deriv = lambda u: (right(u+du) - right(u-du))/(2*du)

    def __call__(self, u):
        # TODO Improve this
        dim = len(u)
        matrix = np.zeros((dim, dim))
        if self._order == 1:
            matrix[:-1, :-1] = np.diag(np.ones(dim-2), k=1) - np.diag(np.ones(dim-2), k=-1)
            matrix[0, :] = matrix[1, :]
            matrix[-1, -1] = 2*self._dx * self._right(u[-1]) / u[-1]
            matrix[-2, -1] = 1
            du = 1/(2*self._dx) * matrix @ u
        elif self._order == 2:
            raise NotImplementedError("Second-order FD not yet implemented")
            du = np.zeros(dim)
            du[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/self._dx**2
            du[0] = du[1]
            du[-1] = 2 * (self._right(u[-1])/self._dx - (u[-1] - u[-2])/self._dx**2)
        return du

