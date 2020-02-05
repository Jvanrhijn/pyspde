from math import sqrt
import copy
from multiprocessing import Process, Queue
from datetime import datetime
import random

import numpy as np
import matplotlib.pyplot as plt

from src.integral import Integral
from src.integrators import Midpoint

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda s, *args, **kwargs: s


class SPDE:

    def __init__(self, linear, da, noise, points, left, right, space_range=(0, 1), right_deriv=None):
        if space_range != (0, 1):
            raise NotImplementedError("Only unit space interval supported")
        self.noise = noise
        self.linear = linear
        self.da = da
        self.space_range = space_range
        self.left = left
        self.right = right
        if right_deriv is not None:
            self.right_deriv = right_deriv
        else:
            self.right_deriv = lambda u: (right(u + 0.001) - right(u))/0.001
        self.points = points


class TrajectorySolver:

    def __init__(self, spde, steps, tmax, initial, linear_solver, integrator=Midpoint()):
        self._spde = spde
        self._tmax = tmax
        self._steps = steps
        self._dt = tmax/steps
        self._dx = (spde.space_range[1] - spde.space_range[0])/spde.points
        self._xs = np.linspace(spde.space_range[0] + self._dx, spde.space_range[1], spde.points)
        self._initial = initial
        self._integrator = integrator
        # linear solver should propagate half time-step for MP algorithm
        self._linsolve_type = linear_solver
        self._linear_solver = linear_solver(spde)
        self._linear_solver.set_timestep(self._dt / integrator.ipsteps())
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
        self._linear_solver.set_timestep(self._dt/2)

    def solve(self, average=False):
        for i in range(self._steps):
            self._solution[i+1] = self._integrator.step(
                self._solution[i], self._spde, self._linear_solver, self._dt, average)

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
            trajectory_solver) for _ in range(ensembles*processes*blocks)]
        self._ensembles = ensembles
        self._blocks = blocks
        self._processes = processes

        # storage
        self._observables = observables
        # hacky way to get proper dimensions for integral observables
        self._block_means = {
            key: (np.zeros((blocks*processes, *trajectory_solver.solution.shape)) if not isinstance(f, Integral)
            else np.zeros((blocks*processes, *(f(trajectory_solver.solution).shape))))
                for key, f in self._observables.items()
        }
        self._block_step_errors = copy.deepcopy(self._block_means)

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

        self.means = {
            key: np.mean(bmean, axis=0) for key, bmean in self._block_means.items()
        }

        self.sample_errors = {
            key: np.std(b, axis=0)/sqrt(max(1, self._blocks-1)) for key, b in self._block_means.items()
        }

        self.step_errors = {
            key: np.mean(step_error, axis=0) for key, step_error in self._block_step_errors.items()
        }

    def _post_process(self, queue):
        start_index = 0
        for _ in range(self._processes):

            # obtain individual block results from queue
            results = queue.get()
            results, fine_results = zip(*results)
            results, fine_results = list(results), list(fine_results)

            num_results = len(results)

            # calculate range in internal trajectory storage
            r = (start_index, start_index + num_results)
            start_index += num_results

            # perform extrapolation to small step size limit
            #order = 1
            #epsilon = 1 / (2**order - 1)
            #true_results = (1 + epsilon) * \
            #    fine_results[:, ::2] - epsilon * results

            # extract block averages per observable
            for res, fres in zip(results, fine_results):
                for name in self._observables:
                    self._block_means[name][r[0]:r[1]] = fres[name][::2]
                    self._block_step_errors[name][r[0]:r[1]] = np.abs(fres[name][::2] - res[name])

    def solve_trajectory(self, queue, threadnr, seed, solvers, verbose):
        results = []
        start_time = datetime.now()
        self._rng.seed(seed)
        # split the solver list into blocks
        blocks = list(self.chunks(solvers, self._blocks))
        trajectory = 0
        for block in self.progress_bar(threadnr)(blocks):
            # Initialize running average for each observable
            block_average = {
                name: 0 for name in self._observables
            }
            block_average_fine = copy.deepcopy(block_average)
            for snr, solver in enumerate(block):
                trajectory += 1
                if verbose:
                    print(
                        f"Thread {threadnr+1}: solving trajectory {trajectory}/{len(solvers)}, \
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
                # average over the block
                # TODO: this formula is numerically unstable, so fix sometime
                for name, f in self._observables.items():
                    block_average[name] += (f(solver.solution) - block_average[name])/(snr+1)
                    block_average_fine[name] += (f(solver_fine.solution) - block_average_fine[name])/(snr+1)
            # put block averages in thread queue
            results.append((block_average, block_average_fine))
        queue.put(results)

    def progress_bar(self, threadnr):
        return tqdm if threadnr == 0 and self._pbar else lambda s, **kwargs: s

    @staticmethod
    def chunks(l, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l[i::n]
