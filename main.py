from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver
from src.noises import WhiteNoise
from src.visualizer import Visualizer
from src.integrators import *
from src.basis import *
from src.deriv import SpectralDerivative

from mpl_toolkits import mplot3d
from examples.potentials import *


if __name__ == "__main__":

    coeff = 1
    points = 15
    steps = 500
    resolution = 2
    tmax = 5
    #blocks = 128
    #samples = 64
    #processes = 4
    blocks = 16
    samples = 4
    processes = 4

    sigma = 0.5
    k = -1

    boundaries = [
        Dirichlet(1),
        Robin(lambda u: (k + 0.5*sigma**2)*u)
        #Robin(lambda u: -sigma*u*np.exp(-u**2))
    ]

    lattice = Lattice(0, 1, points, boundaries)
    basis = FiniteElementBasis(lattice, boundaries)
    #basis = SpectralBasis(lattice, boundaries)

    u0 = np.ones((1, points))

    noise = WhiteNoise(1) 

    d1 = DerivativeOperator(1, lattice, boundaries)

    multinoise_volatility = lambda u: sigma*sqrt(2)*np.sqrt(1 + u**2)
    multinoise_drift = lambda u: -d1(u)**2 * u/(1 + u**2) - (k - sigma**2)**2 * u / (1 + u**2) \
        + 4*sigma**2 * u 

    multinoise_drift_backtrans = lambda u: -u/(1 + u**2) * d1(u)**2 - (k**2 + 0.25*sigma**4) * (1 - u**2/(1 + u**2))*u \
        + 4*sigma**2 * u

    gaussian_volatility = lambda u: sqrt(2) * sigma * np.exp(-u**2/2)
    gaussian_drift = lambda u: u*d1(u)**2 + sigma**4 * np.exp(-2*u**2) * (u**3 - 1.5*u) + 2 * sigma**2 * -2*u * np.exp(-u**2)

    spde = SPDE(
        coeff,
        lambda u: -d1(u)**2 / u + 4*sigma**2*u,
        lambda u: sqrt(2)*u*sigma,
        noise
    )

    problem = StochasticPartialProblem(
        spde,
        boundaries,
        lattice
    )

    #stepper = MidpointFEM(lattice, basis, problem, solver=lambda f, x: opt.broyden1(f, x, iter=2))
    stepper = ThetaScheme(1, lattice, basis, problem)
    #stepper = MidpointIP(SpectralSolver(problem))
    #stepper = RK4IP(SpectralSolver(problem))
    #stepper = DifferentialWeakMethod(lattice, problem)

    solver = TrajectorySolver(problem, steps, tmax, u0, stepper, resolution=resolution)

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

    print(f"Max step error =   {step_errors[:, -1].max()}")
    print(f"Max sample error = {sample_errors[:, -1].max()}")

    vis = Visualizer(mean, (0, tmax), lattice,
                     sample_error=ensemble_solver.sample_errors["value"][field],
                     step_error=ensemble_solver.step_errors["value"][field])

    vis2 = Visualizer(square, (0, tmax), lattice,
                      sample_error=ensemble_solver.sample_errors["square"][field],
                      step_error=ensemble_solver.step_errors["square"][field])

    mesh_points = min([30, points, steps])
    fig, ax = vis.surface(cstride=points//mesh_points,
                          rstride=int(steps/resolution)//mesh_points)

    fig2, ax2 = vis.steady_state(
        label="Numerical solution", marker='o', linestyle='-.')

    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")


    """Obtain the steady-state solution numerically"""
    """
    from scipy.integrate import solve_bvp
    from scipy.interpolate import interp1d

    velocity = ensemble_solver.means["velocity"][field][-1]
    velocity_squared = ensemble_solver.means["velocitysq"][field][-1]

    vvar = velocity_squared - velocity**2
    # extrapolate to get variance on extended lattice
    xs = np.linspace(0, 1, 100)
    ys = np.ones((2, xs.size))

    v = interp1d(lattice.points, vvar, fill_value="extrapolate")

    def du(t, u):
        mesh = np.linspace(0, 1, len(u[1]))
        return np.vstack((u[1], u[1]**2 + v(mesh)))
    
    
    def bc(ua, ub):
        return np.array([ua[0] - boundaries[0](), ub[0] - boundaries[1]()])
    
    
    res = solve_bvp(du, bc, xs, ys)


    ax2.plot(xs, 
            res.sol(xs)[0],
            label=r"Analytical solution")

    ax2.legend()
    """
    ax2.plot(ts, np.exp(k*ts), label="Analytical solution")

    fig, ax = vis2.surface(cstride=points//mesh_points,
                          rstride=int(steps/resolution)//mesh_points)

    fig3, ax3 = vis2.steady_state(
        'o', label="Numerical solution", marker='o', linestyle='-.')
    ax3.plot(ts, np.exp((2*k + sigma**2)*ts), label="Geometric Brownian Motion")
    ax3.plot(ts, 
    (boundaries[0]()**2 + sigma**2/(2*k + sigma**2))*np.exp((2*k+sigma**2)*ts) - sigma**2/(2*k + sigma**2),
    label="Multinoise")
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.legend()

    #taus = vis.taxis
    #fig, ax = vis.at_origin()
    #ax.plot(taus, np.exp(k*taus), linestyle='-.')

    plt.show()
