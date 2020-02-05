import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, solution, trange, srange, sample_error=None, step_error=None):
        self._solution = solution
        self._sample_error = sample_error
        self._step_error = step_error
        self._trange = trange
        self._srange = srange
        steps, points = solution.shape
        dx = (srange[1] - srange[0])/points
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

    @property
    def taxis(self):
        return self._ts