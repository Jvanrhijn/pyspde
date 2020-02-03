from math import sqrt
import copy
from multiprocessing import Process, Queue
from datetime import datetime
import random

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
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
        self._tmax = tmax
        self._steps = steps
        self._dt = tmax/steps
        self._dx = 1/spde.points
        self._xs = np.linspace(self._dx, 1, spde.points)
        self._initial = initial
        # linear solver should propagate half time-step for MP algorithm
        self._linsolve_type = linear_solver
        self._linear_solver = linear_solver(spde, self._dt/2)
        self._solution = np.zeros((steps+1, spde.points))
        self._solution[0] = initial
        self._time = 0

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        self._steps = steps
        self._dt = self._tmax / steps
        initial = self._solution[0]
        self._solution = np.zeros((steps+1, self._spde.points))
        self._solution[0] = initial
        self._linear_solver = self._linsolve_type(self._spde, self._dt/2)

    def solve(self, average=False):
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
            w = noise(self._time + 0.5*self._dt, average=average)
            for _ in range(midpoints_iters):
                a = a0 + 0.5*self._dt * \
                    self._spde.da(a, self._time + 0.5*self._dt, w)
            # propagate solution to t + dt
            if self._spde.linear != 0.0:
                self._solution[i +
                               1] = self._linear_solver.propagate_step(2*a - a0)
            else:
                self._solution[i+1] = 2*a - a0
            self._time += self._dt

    @property
    def solution(self):
        return self._solution

    @property
    def seed(self):
        return self._spde.noise.seed

    @seed.setter
    def seed(self, seed):
        self._spde.noise.seed = seed


class EnsembleSolver:

    def __init__(self, trajectory_solver, ensembles, processes=1, verbose=True, pbar=False, seed=None):
        self._verbose = verbose
        self._pbar = pbar
        self._rng = random.Random(seed)
        if verbose and pbar:
            raise ValueError(
                "EnsembleSolver can't output logging data and progress bar")
        self._trajectory_solvers = [copy.deepcopy(
            trajectory_solver) for _ in range(ensembles)]
        self._ensembles = ensembles
        self._storage = np.zeros(
            (self._ensembles, *trajectory_solver.solution.shape))
        self.mean = None
        self.square = None
        self.sample_error = None
        self.step_error = np.zeros(self._storage.shape)
        self.square_step_error = None
        self.square_sample_error = None
        self._processes = processes

    def solve(self):
        queue = Queue()
        # split the solver list into chunks
        solver_chunks = list(self.chunks(
            self._trajectory_solvers, self._processes))
        seeds = [self._rng.randint(0, 2**32-1) for _ in range(self._processes)]
        for thread, chunk in enumerate(solver_chunks):
            p = Process(target=self.solve_trajectory,
                        args=(queue, thread, seeds[thread], chunk, self._verbose))
            p.start()
        start_index = 0
        for chunk in solver_chunks:
            # obtain individual trajectory results from queue
            results = queue.get()
            results, fine_results = zip(*results)
            results, fine_results = np.array(results), np.array(fine_results)
            num_results = len(results)
            # calculate range in internal trajectory storage
            r = (start_index, start_index + num_results)
            start_index += num_results
            # perform extrapolation to small step size limit
            order = 1
            epsilon = 1 / (2**order - 1)
            true_results = (1 + epsilon) * \
                fine_results[:, ::2] - epsilon * results
            # store results
            self._storage[r[0]:r[1]] = true_results
            self.step_error[r[0]:r[1]] = np.abs(fine_results[:, ::2] - results)

        # calculate some moments and errors
        self.step_error = np.mean(self.step_error, axis=0)
        self.mean = np.mean(self._storage, axis=0)
        self.square_step_error = 2*self.mean*self.step_error + self.step_error**2
        self.square = np.mean(self._storage**2, axis=0)
        self.sample_error = np.std(
            self._storage, axis=0)/sqrt(self._ensembles - 1)
        self.square_sample_error = np.std(
            self._storage**2, axis=0)/sqrt(self._ensembles - 1)

    def solve_trajectory(self, queue, threadnr, seed, solvers, verbose):
        results = []
        start_time = datetime.now()
        self._rng.seed(seed)
        for snr, solver in self.progress_bar(threadnr)(enumerate(solvers), total=len(solvers)):
            if verbose:
                print(
                    f"Thread {threadnr+1}: solving trajectory {snr+1}/{len(solvers)}, time = {datetime.now() - start_time}")
            # for time step error estimation, copy the solver with a finer
            # step size
            solver_fine = copy.deepcopy(solver)
            solver_fine.steps = solver.steps * 2
            # generate seed
            seed = self._rng.randint(0, 2**32-1)
            solver.seed = seed
            solver_fine.seed = seed
            # solve both the original and the fine trajectory
            # 'average' keyword telse the coarse solver to average
            # two subsequent fine noise terms for consisteny
            solver.solve(average=True)
            solver_fine.solve(average=False)
            results.append((solver.solution, solver_fine.solution))
        queue.put(results)

    def progress_bar(self, threadnr):
        return tqdm if threadnr == 0 and self._pbar else lambda s, **kwargs: s

    @staticmethod
    def chunks(l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]


class Visualizer:

    def __init__(self, solution, trange, srange, sample_error=None, step_error=None):
        self._solution = solution
        self._sample_error = sample_error
        self._step_error = step_error
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
        ax.plot_surface(xs, ts, self._solution, *
                        args, cmap="viridis", **kwargs)
        return fig, ax

    def steady_state(self, *args, **kwargs):
        ss = self._solution[-1]
        fig, ax = plt.subplots(1)
        if self._step_error is not None:
            step_error = self._step_error[-1]
            ax.errorbar(self._xs, ss, yerr=step_error, **kwargs)
        else:
            step_error = 0
            ax.plot(self._xs, ss, *args, **kwargs)
        if self._sample_error is not None:
            error = self._sample_error[-1] + step_error
            ax.fill_between(self._xs, ss-error, ss+error, alpha=0.25)
        ax.grid(True)
        return fig, ax

    def at_origin(self, *args, **kwargs):
        y = self._solution[:, 0]
        fig, ax = plt.subplots(1)
        if self._step_error is not None:
            step_error = self._step_error[:, 0]
            ax.errorbar(self._ts, y, yerr=step_error, **kwargs)
        else:
            step_error = 0
            ax.plot(self._ts, y, *args, **kwargs)
        if self._sample_error is not None:
            error = self._sample_error[:, 0] + step_error
            ax.fill_between(self._ts, y-error, y+error, alpha=0.25)
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

    def __call__(self, u):
        # TODO Improve this
        dim = len(u)
        matrix = np.zeros((dim, dim))
        if self._order == 1:
            matrix[:-1, :-1] = np.diag(np.ones(dim-2),
                                       k=1) - np.diag(np.ones(dim-2), k=-1)
            matrix[0, :] = matrix[1, :]
            matrix[-1, -1] = 2*self._dx * self._right(u[-1]) / u[-1]
            matrix[-2, -1] = 1
            derivative = 1/(2*self._dx) * matrix @ u
        elif self._order == 2:
            raise NotImplementedError("Second-order FD not yet implemented")
            derivative = np.zeros(dim)
            derivative[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/self._dx**2
            derivative[0] = derivative[1]
            derivative[-1] = 2 * (self._right(u[-1])/self._dx -
                                  (u[-1] - u[-2])/self._dx**2)
        return derivative
