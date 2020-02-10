from math import sqrt
import copy
from multiprocessing import Process, Queue
from datetime import datetime
import random
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from src.integral import Integral
from src.integrators import Midpoint

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda s, *args, **kwargs: s


class Boundary(Enum):

    DIRICHLET = 1
    ROBIN = 2

class BoundaryCondition(ABC):

    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def kind(self):
        pass


# TODO: generalize to time-dependent BC
class Dirichlet(BoundaryCondition):

    def __init__(self, value):
        self._value = value

    def __call__(self, *args):
        return self._value

    def kind(self):
        return Boundary.DIRICHLET


# TODO: generalize to time-dependent BC
class Robin(BoundaryCondition):

    def __init__(self, value):
        self._value = value

    def __call__(self, function, *args):
        return self._value(function)

    def kind(self):
        return Boundary.ROBIN


class Lattice:

    def __init__(self, origin, end, points, boundaries):
        if origin != 0 or end != 1:
            raise NotImplementedError("Only unit spatial interval supported")
        include_left = not boundaries[0].kind() == Boundary.DIRICHLET
        include_right = not boundaries[1].kind() == Boundary.DIRICHLET
        self._increment = (end - origin)/points
        if include_left and include_right:
            self._points = np.arange(origin, end + self._increment, self._increment)
        elif include_left and not include_right:
            self._points = np.arange(origin, end, self._increment)
        elif not include_left and include_right:
            self._points = np.arange(origin + self._increment, end + self._increment, self._increment)
        elif not include_left and not include_right:
            self._points = np.arange(origin + self._increment, end, self._increment)
        self._midpoints = np.arange(self._increment/2, end, self._increment)

    @property 
    def points(self):
        return self._points

    @property
    def increment(self):
        return self._increment

    @property
    def midpoints(self):
        return self._midpoints

    @property
    def range(self):
        return (self._points[0], self._points[-1])

class SPDE:

    def __init__(self, linear, drift, volatility, noise):
        self.linear = linear
        self.drift = drift
        self.volatility = volatility
        self.noise = noise

    
class StochasticPartialProblem:

    def __init__(self, spde, boundaries, lattice):
        self._spde = spde
        self._lattice = lattice
        if len(boundaries) != 2:
            raise NotImplementedError("Too many boundary conditions: only 1D SPDE supported")
        self.left = boundaries[0]
        self.right = boundaries[1]

    @property
    def spde(self):
        return self._spde

    @property
    def lattice(self):
        return self._lattice


class TrajectorySolver:

    def __init__(self, problem, steps, tmax, initial, stepper):
        self._problem = problem
        self._tmax = tmax
        self._steps = steps
        self._dt = tmax/steps
        self._initial = initial
        self._stepper = stepper
        self._solution = np.zeros((1, steps+1, len(problem.lattice.points)))
        self._solution[:, 0, :] = initial
        self._time = 0

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        self._steps = steps
        self._dt = self._tmax / steps
        initial = self._solution[:, 0, :]
        self._solution = np.zeros((1, steps+1, len(self._problem.lattice.points)))
        self._solution[:, 0, :] = initial
        self._stepper.set_timestep(self._dt)

    def solve(self, average=False):
        for i in range(self._steps):
            self._solution[:, i+1] = self._stepper.step(
                self._solution[:, i], self._problem,  average)

    @property
    def solution(self):
        return self._solution

    @property
    def seed(self):
        return self._problem.spde.noise.seed

    @seed.setter
    def seed(self, seed):
        self._problem.spde.noise.seed = seed


class EnsembleSolver:

    def __init__(self, trajectory_solver, ensembles, observables=dict(),
                 blocks=2, processes=1, verbose=True, pbar=False, seed=None, check=True):
        self._check = check
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
                    self._block_means[name][r[0]:r[1]] = fres[name][:, ::2] if self._check else res[name]
                    if self._check:
                        self._block_step_errors[name][r[0]:r[1]] = np.abs(fres[name][:, ::2] - res[name])

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
                if self._check:
                    solver_fine = copy.deepcopy(solver)
                    solver_fine.steps = solver.steps * 2
                # solve both the original and the fine trajectory
                # 'average' keyword telse the coarse solver to average
                # two subsequent fine noise terms for consisteny
                solver.solve(average=self._check)
                if self._check:
                    solver_fine.solve(average=False)
                # average over the block
                # TODO: this formula is numerically unstable, so fix sometime
                for name, f in self._observables.items():
                    block_average[name] += (f(solver.solution) - block_average[name])/(snr+1)
                    if self._check:
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
