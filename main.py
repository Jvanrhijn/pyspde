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
    steps = 1000
    tmax = 5
    blocks = 10
    samples = 1
    processes = 4
    space_range = (0.0, 1.0)

    sigma = 0.1
    k = -1

    fields = 2

    f = 1
    #g = lambda u: (k - sigma**2)*u
    g = lambda u: k*u
    gderiv = lambda u: k
    #g = lambda u: 0
    #g = lambda u: k*u
    #gderiv = lambda u: k - sigma**2
    #g = lambda u: -0.5*sigma*u*np.exp(-u**2)
    #gderiv = None

    u0 = np.ones((fields, points))
    noise = WhiteNoise(1, points, fields=fields)

    d1 = DerivativeOperator(1, 1/points, f, g)

    #da = linmult_arnold(points, k, sigma, f, g)
    #da = lambda a, t, w: sigma*w
    #da = geometric_brownian(points, k, sigma, f, g)
    #da = gaussian_arnold(points, k, sigma, f, g)
    da = linear(points, k, sigma, f, g)
    #da = lambda a, t, w: k*a + sigma*a*w

    def da(a, t, w):
        field1 = a[0]
        field2 = a[1]
        return np.array([
            linear(points, k, sigma, f, g)(a[0], t, w[0]),
            geometric_brownian(points, k, sigma, f, g)(a[1], t, w[1]),
        ])

    spde = SPDE(coeff, da,
                noise, points, f, g, right_deriv=gderiv, space_range=space_range)

    #solver = TrajectorySolver(spde, steps, tmax, u0, lambda *args: SpectralSolver(*args, store_midpoint=True))
    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver, fields=fields, integrator=Midpoint())

    ts = np.linspace(space_range[0] + 1/points, space_range[1], points)

    ensemble_solver = EnsembleSolver(
            solver, 
            samples, 
            observables={
                "value": lambda x: x,
                "square": lambda x: x**2,
                #"energy": Integral(lambda x: 0.5*x**2, ts, 1),
            },
            blocks=blocks, 
            processes=processes, 
            verbose=True, 
            pbar=False, 
            seed=1
    )

    ensemble_solver.solve()
    field = 1

    mean = ensemble_solver.means["value"][field]
    square = ensemble_solver.means["square"][field]

    step_errors = ensemble_solver.step_errors["value"]
    sample_errors = ensemble_solver.sample_errors["value"]

    print(f"Max step error =   {step_errors.max()}")
    print(f"Max sample error = {sample_errors.max()}")

    vis = Visualizer(mean, (0, tmax), space_range,
                     sample_error=ensemble_solver.sample_errors["value"][field],
                     step_error=ensemble_solver.step_errors["value"][field])
    vis2 = Visualizer(square, (0, tmax), space_range,
                      sample_error=ensemble_solver.sample_errors["square"][field],
                      step_error=ensemble_solver.step_errors["square"][field])

    mesh_points = min([30, points, steps])
    fig, ax = vis.surface(cstride=points//mesh_points,
                          rstride=steps//mesh_points)
    #fig, ax = vis.surface(cstride=1,
    #                      rstride=30)

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
    #         #f**2*np.exp(2*k*ts),
    #         #f*np.exp((2*k + sigma**2)*ts),
             (f + sigma**2/(2*k+sigma**2)) *
             np.exp((2*k + sigma**2)*ts) - sigma**2/(2*k + sigma**2),
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
