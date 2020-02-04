from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver
from src.noises import WhiteNoise
from src.visualizer import Visualizer
from src.integrators import *

from mpl_toolkits import mplot3d
from examples.potentials import *


if __name__ == "__main__":

    coeff = 1
    points = 10
    steps = 100
    tmax = 5
    blocks = 80
    samples = blocks*4
    processes = 4
    space_range = (0.0, 1.0)

    sigma = 0.5
    k = -1

    f = 1
    #g = lambda u: (k - 0.5*sigma**2)*u
    #g = lambda u: k*u
    #gderiv = lambda u: k - sigma**2
    g = lambda u: -0.5*sigma*u*np.exp(-u**2)
    #gderiv = None

    u0 = np.ones(points)
    noise = WhiteNoise(2, points)

    #da = linmult_arnold(points, k, sigma, f, g)
    #da = lambda a, t, w: sigma*w
    #da = linear(points, k, sigma, f, g)
    da = gaussian_arnold(points, k, sigma, f, g)

    spde = SPDE(coeff, da,
                noise, points, f, g, space_range=space_range)

    #solver = TrajectorySolver(spde, steps, tmax, u0, lambda *args: SpectralSolver(*args, store_midpoint=True))
    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver, integrator=Midpoint())

    ts = np.linspace(space_range[0] + 1/points, space_range[1], points)

    ensemble_solver = EnsembleSolver(
            solver, 
            samples, 
            observables={
                "value": lambda x: x,
                "square": lambda x: x**2,
                "energy": Integral(lambda x: 0.5*x**2, ts, 2),
            },
            blocks=blocks, 
            processes=processes, 
            verbose=False, 
            pbar=True, 
            seed=0
    )

    ensemble_solver.solve()

    mean = ensemble_solver.means["value"]
    square = ensemble_solver.means["square"]

    plt.figure()
    plt.plot(ensemble_solver._storage[0, -1, :])

    step_errors = ensemble_solver.step_errors["value"]
    sample_errors = ensemble_solver.sample_errors["value"]

    print(f"Max step error =   {step_errors.max()}")
    print(f"Max sample error = {sample_errors.max()}")

    #plt.figure()
    #plt.plot(np.linspace(0, tmax, steps+1), ensemble_solver.means["energy"])

    vis = Visualizer(mean, (0, tmax), space_range,
                     sample_error=ensemble_solver.sample_errors["value"],
                     step_error=ensemble_solver.step_errors["value"])
    vis2 = Visualizer(square, (0, tmax), space_range,
                      sample_error=ensemble_solver.sample_errors["square"],
                      step_error=ensemble_solver.step_errors["square"])

    mesh_points = min([10, points, steps])
    fig, ax = vis.surface(cstride=points//mesh_points,
                          rstride=steps//mesh_points)
    ts = vis.xaxis

    fig2, ax2 = vis.steady_state(
        label="Numerical solution", marker='o', linestyle='-.')
    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")
    ax2.plot(ts, np.exp(k*ts),
             label=r"Analytical; $B = \sigma\sqrt{1 + \phi^2}$")

    ax2.legend()

    fig3, ax3 = vis2.steady_state(
        'o', label="Numerical solution", marker='o', linestyle='-.')
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.plot(ts,
             f**2*np.exp(2*k*ts),
             #f*np.exp((2*k + sigma**2)*ts),
             #(f + sigma**2/(2*k+sigma**2)) *
             #np.exp((2*k + sigma**2)*ts) - sigma**2/(2*k + sigma**2),
             label=r"Analytical, $B = \sigma\sqrt{1 + \phi^2}$")

#
#    """Now, let's compute the steady-state solution using Scipy"""
#    from scipy.integrate import solve_bvp
#
#    def du(t, u):
#        return np.vstack((u[1], u[0] / (1 + u[0]**2)*(u[1]**2 + (k - sigma**2)**2)))
#        #return np.vstack((u[1], u[0]*u[1]**2 + sigma**2 / 4 * np.exp(-2*u[0]**2) * (u[0]**3 - 1.5*u[0])))
#
#    def bc(ua, ub):
#        return np.array([ua[0] - f, ub[1] - g(ub[0])])
#
#    xs = np.linspace(0, 1, 100)
#    ys = np.zeros((2, xs.size))
#    res = solve_bvp(du, bc, xs, ys)
#    ax2.plot(xs, res.sol(xs)[0], label="Scipy solution")
#    ax3.plot(xs, res.sol(xs)[0]**2, label="Scipy solution")
#    """End Scipy solution"""
#
    plt.show()
