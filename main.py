from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver
from src.noises import WhiteNoise
from src.visualizer import Visualizer
from src.integrators import *
from src.basis import *

from mpl_toolkits import mplot3d
from examples.potentials import *


if __name__ == "__main__":

    coeff = 1
    points = 10
    steps = 1000
    tmax = 1
    blocks = 16
    samples = 4
    processes = 4

    sigma = 0.5
    k = -1

    boundaries = [
        Dirichlet(0.0),
        Dirichlet(0.0),
        #Robin(lambda u: (k -.5*sigma**2)*u)
    ]

    lattice = Lattice(-0.5, 0.5, points, boundaries)
    basis = FiniteElementBasis(lattice, boundaries)

    u0 = np.ones((1, points))*0
    noise = WhiteNoise(1, points+1) 

    d1 = DerivativeOperator(1, lattice, boundaries)

    spde = SPDE(
        coeff,
        lambda u: 0.0,
        lambda u: sigma*sqrt(2),
        noise
    )

    problem = StochasticPartialProblem(
        spde,
        boundaries,
        lattice
    )

    stepper = ThetaScheme(1, lattice, basis, tmax/steps)
    #stepper = Midpoint(SpectralSolver(problem), tmax/steps)
    #stepper = RK4(SpectralSolver(problem), tmax/steps)

    solver = TrajectorySolver(problem, steps, tmax, u0, stepper)

    ts = lattice.points

    ensemble_solver = EnsembleSolver(
            solver, 
            samples, 
            observables={
                "value": lambda x: x,
                "square": lambda x: x**2,
            },
            blocks=blocks, 
            processes=processes, 
            verbose=True, 
            pbar=False, 
            seed=1,
            check=True,
    )

    ensemble_solver.solve()
    field = 0

    mean = ensemble_solver.means["value"][field]
    square = ensemble_solver.means["square"][field]

    step_errors = ensemble_solver.step_errors["value"]
    sample_errors = ensemble_solver.sample_errors["value"]

    print(f"Max step error =   {step_errors.max()}")
    print(f"Max sample error = {sample_errors.max()}")

    vis = Visualizer(mean, (0, tmax), lattice,
                     sample_error=ensemble_solver.sample_errors["value"][field],
                     step_error=ensemble_solver.step_errors["value"][field])

    vis2 = Visualizer(square, (0, tmax), lattice,
                      sample_error=ensemble_solver.sample_errors["square"][field],
                      step_error=ensemble_solver.step_errors["square"][field])

    mesh_points = min([30, points, steps])
    fig, ax = vis.surface(cstride=points//mesh_points,
                          rstride=steps//mesh_points)

    fig2, ax2 = vis.steady_state(
        label="Numerical solution", marker='o', linestyle='-.')

    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")
    #ax2.plot(ts, np.exp(k*ts),
    #         label=r"Analytical solution")

    ax2.legend()

    fig, ax = vis2.surface(cstride=points//mesh_points,
                          rstride=steps//mesh_points)

    fig3, ax3 = vis2.steady_state(
        'o', label="Numerical solution", marker='o', linestyle='-.')
    ax3.plot(ts, sigma**2*(ts+0.5)*(1 - (ts+0.5)))
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    #ax3.plot(ts,
    #         np.exp((2*k + sigma**2)*ts),
    #         label=r"Analytical solution")

    plt.show()
