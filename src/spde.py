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

    def __init__(self, trajectory_solver, ensembles, observables=dict(), 
                    blocks=2, processes=1, verbose=True, pbar=False, seed=None):
        self._verbose = verbose
        self._pbar = pbar
        self._rng = random.Random(seed)
        if verbose and pbar:
            raise ValueError(
                "EnsembleSolver can't output logging data and progress bar")
        self._trajectory_solvers = [copy.deepcopy(
            trajectory_solver) for _ in range(ensembles)]
        self._ensembles = ensembles

        # storage
        self._observables = observables
        self._storage = np.zeros(
            (self._ensembles, *trajectory_solver.solution.shape))
        self._observe_data = {
            key: f(self._storage) for key, f in self._observables.items()
        }

        self._blocks = blocks
        self.step_error = np.zeros(self._storage.shape)
        self._processes = processes

    def solve(self):
        queue = Queue()
        # split the solver list into chunks for threading
        solver_chunks = self.chunks(
            self._trajectory_solvers, self._processes)

        # generate seeds for each thread
        seeds = [self._rng.randint(0, 2**32-1) for _ in range(self._processes)]

        # dispatch threads
        for thread, chunk in enumerate(solver_chunks):
            p = Process(target=self.solve_trajectory,
                        args=(queue, thread, seeds[thread], chunk, self._verbose))
            p.start()

        # retrieve the data from the queue and post-process
        self._post_process(queue)

        # do blocking
        blocks = {
            key: list(self.chunks(data, self._blocks)) for key, data in self._observe_data.items()
        }

        block_means = {
            key: np.mean(block, axis=0) for key, block in blocks.items()
        }

        self.means = {
            key: np.mean(bmean, axis=0) for key, bmean in block_means.items()
        }

        self.sample_errors = {
            key: np.std(b, axis=0)/sqrt(len(b)-1) for key, b in self.means.items()
        }

        self.step_errors = {
            key: np.mean(np.abs((f(self.means[key] + 0.001) - f(self.means[key]))/0.001)*self.step_error, axis=0) 
                for key, f in self._observables.items()
        }

    def _post_process(self, queue):
        start_index = 0
        for _ in range(self._processes):

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
            for key, f in self._observables.items():
                self._observe_data[key][r[0]:r[1]] = f(true_results)

    def solve_trajectory(self, queue, threadnr, seed, solvers, verbose):
        results = []
        start_time = datetime.now()
        self._rng.seed(seed)
        for snr, solver in self.progress_bar(threadnr)(enumerate(solvers), total=len(solvers)):
            if verbose:
                print(
                    f"Thread {threadnr+1}: solving trajectory {snr+1}/{len(solvers)}, \
                        time = {datetime.now() - start_time}")
            # generate seed
            seed = self._rng.randint(0, 2**32-1)
            solver.seed = seed
            # for time step error estimation, copy the solver and set a finer
            # step size
            solver_fine = copy.deepcopy(solver)
            solver_fine.steps = solver.steps * 2
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
