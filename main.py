from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver, SpectralPeriodic
from src.noises import WhiteNoise, MollifiedWhiteNoise, GaussianMollifiedNoise
from src.visualizer import Visualizer
from src.integrators import *
from src.basis import *
from src.deriv import *

from math import log, pi

from mpl_toolkits import mplot3d


if __name__ == "__main__":

    coeff = 1
    points = 60
    dx = 1/points
    tmax = 2
    steps = 1000
    resolution = 2
    blocks = 4
    samples = 4
    processes = 4

    sigma = 0.5
    k = -1

    boundaries = [
        #Dirichlet(0),
        #Dirichlet(0),
        Dirichlet(1),
        Robin(lambda u: (k-sigma**2)*u)
    ]

    lattice = Lattice(-1, 1, points, boundaries)
    basis = FiniteElementBasis(lattice, boundaries)
    #basis = SpectralBasis(lattice, boundaries)

    d2 = LamShinDerivativeSquared(lattice)
    dc1 = CentralDifference(lattice)
    db1 = BackwardDifference(lattice)
    lapl = Laplacian(lattice, boundaries)

    #u0 = np.zeros(lattice.points.shape)
    u0 = np.exp(k*(lattice.points - lattice.range[0]))

    #noise = WhiteNoise(1) 
    # compute mollifier (normalized)
    eps = 2*dx
    #m = lambda x: 1/eps * np.exp(-x**2/(2*eps**2))
    #m = lambda x: 1/eps * np.exp(1 - 1/(1 - (np.abs(x/eps) - 0.01)**2))
    def m(x):
        return np.piecewise(x,
        [
            np.abs(x) < 1,
        ],
        [
            lambda x: 1/eps * np.exp(-1/(1 - (x/eps)**2)),
            0
        ])

    m = lambda x: 1/sqrt(2*pi*eps**2) * np.exp(-x**2/(2*eps**2))
    
    moll = m(lattice.points)
    moll /= np.trapz(moll, lattice.points)

    # mollified heat kernel
    ts_fine, h = np.linspace(*lattice.range, points*100, retstep=True)
    moll_fine = m(ts_fine)
    moll_fine /= np.trapz(moll_fine, ts_fine)
    heat_kernel = lambda x: -0.5*x*np.sign(x)
    moll_hk = np.convolve(moll_fine, heat_kernel(ts_fine)*h, mode="same")

    # renormalization constants
    c = np.trapz(moll_hk*moll_fine, ts_fine)
    d_hk = lambda x: -0.5*np.sign(x)
    moll_dhk = np.convolve(d_hk(ts_fine), moll_fine*h, mode="same")
    #cbar = np.trapz(moll_dhk**2, ts_fine)
    cbar = -c/2

    print(c, cbar)

    #noise = MollifiedWhiteNoise(1, moll) 
    noise = GaussianMollifiedNoise(eps)
    #noise = WhiteNoise(1)

    def discrete_product(x, y):
        prod = np.zeros(x.shape)
        a = 1
        b = 0.5
        prod[:, 1:-1] = 1/(2*(a + b)) * (a*x[:, 1:-1] * y[:, 1:-1] \
            + b*(x[:, 1:-1]*y[:, 2:] + x[:, 2:]*y[:, 1:-1]) \
            + a*x[:, 2:]*y[:, 2:])
        # extrapolate boundary points
        prod[:, 0] = 2*prod[:, 1] - prod[:, 2]
        prod[:, -1] = 2*prod[:, -2] - prod[:, -3]
        return prod

    # second term in drift is Stratonovich -> Ito correction
    # rest are renormalization terms
    f = lambda u: -1/u
    #f = lambda u: -u / (1 + u**2)
    g = lambda u: sqrt(2)*u*sigma
    #g = lambda u: sqrt(2)*sigma*np.sqrt(1 + u**2)
    gprime = lambda u: sqrt(2)*sigma
    #gprime = lambda u: sqrt(2)*sigma *  u / np.sqrt(1 + u**2)

    spde = SPDE(
        coeff,
        #lambda u, t: 0,
        lambda u, t: f(u) * (dc1(u)**2 - g(u)**2*cbar) - g(u)*gprime(u)*c,
        lambda u, t: 0,
        #lambda u, t: sqrt(2)*sigma,
        lambda u, t: g(u),
        noise
    )

    problem = StochasticPartialProblem(
        spde,
        boundaries,
        lattice
    )

    #stepper = ThetaScheme(1, lattice, basis, problem)
    #stepper = MidpointIP(SpectralSolver(problem))
    #stepper = RK4IP(GalerkinSolver(problem, basis))
    stepper = MidpointIP(GalerkinSolver(problem, basis))

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
        label="Numerical solution", marker='.', linestyle='-.')

    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")

    t0 = lattice.range[0]
    ax2.plot(ts, np.exp(k*(ts - t0)), label="Analytical solution")

    #ax2.plot(ts, 0.5*(ts - t0)*(1-0.5*(ts - t0)))

    fig, ax = vis2.surface(cstride=points//mesh_points,
                          rstride=int(steps/resolution)//mesh_points)

    fig3, ax3 = vis2.steady_state(
        '.', label="Numerical solution", marker='.', linestyle='-.')
    ax3.plot(ts, np.exp((2*k + sigma**2)*(ts - t0)), label="Geometric Brownian Motion")
    ax3.plot(ts, np.exp(2*k*(ts - t0)), label="Linear Diffusion")
    ax3.plot(ts, 
    (boundaries[0]()**2 + sigma**2/(2*k + sigma**2))*np.exp((2*k+sigma**2)*(ts - t0)) - sigma**2/(2*k + sigma**2),
    label="Multinoise")
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.legend()

    #taus = vis.taxis
    #fig, ax = vis.at_origin()
    #ax.plot(taus, np.exp(k*taus), linestyle='-.')

    plt.show()
