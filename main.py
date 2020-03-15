from math import log, pi, sqrt, exp

from mpl_toolkits import mplot3d

from pyspde import *

if __name__ == "__main__":

    coeff = 1
    points = 31
    tmax = 1

    # CONJECTURE:
    # need to keep steps large relative to points to obtain convergence to
    # the "canonical" solution
    # parabolic scaling: dt = k*dx^2, for some k < 1
    # so when (dt, dx) leads to a "good" solution, a refinement
    # dx -> a*dx requires dt -> a^2 * dt to obtain the same
    # solution again.
    steps = points**2
    resolution = 10
    print(steps)

    #equilibrate_over = (steps//resolution)//2 # take last 50% of time steps as equilibration
    equilibrate_over = 1
    blocks = 32
    samples = 128
    processes = 4

    sigma = sqrt(0.5)
    k = -1

    boundaries = [
        #Dirichlet(0),
        #Dirichlet(0)
        Dirichlet(1),
        Robin(lambda u: (k - 0*sigma**2/2)*u)
    ]

    lattice = Lattice(-0.5, 0.5, points, boundaries)
    dx = lattice.increment

    basis = FiniteElementBasis(lattice, boundaries)
    #basis = SpectralBasis(lattice, boundaries)

    d2 = LamShinDerivativeSquared(lattice)
    dc1 = CentralDifference(lattice)
    db1 = BackwardDifference(lattice)
    lapl = Laplacian(lattice, boundaries)

    #u0 = np.ones(lattice.points.shape)
    u0 = np.exp(k*(lattice.points - lattice.range[0]))

    # compute mollifier (normalized)
    eps = 2*dx

    # mollifier: Gaussian
    m = lambda x: 1/sqrt(2*pi*eps**2) * np.exp(-x**2/(2*eps**2))
    moll = m(lattice.points)
    
    #
    ts, dt = np.linspace(*lattice.range, 10000, retstep=True)
    moll_fine = m(ts)

    # mollified heat kernel
    heat_kernel = lambda x: -0.5*x*np.sign(x)
    moll_hk = np.convolve(moll_fine, heat_kernel(ts)*dt, mode="same")

    # renormalization constants
    c = np.trapz(moll_hk*moll_fine, ts) * 0
    d_hk = lambda x: -0.5*np.sign(x)
    moll_dhk = np.convolve(d_hk(ts), moll_fine*dt, mode="same")
    #cbar = np.trapz(moll_dhk**2, ts)
    #cbar =  0.5 + c
    cbar = -c/2

    print(c, cbar)

    noise = FourierMollifiedNoise(m, lattice)
    #noise = MollifiedWhiteNoise(1, moll)
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

    #f = lambda u: -1/u
    #f = lambda u: -u / (1 + u**2)
    f = lambda u: 0
    #h = lambda u: 0
    #h = lambda u: -(k**2 + sigma**4/4) * (1 - u**2 / (1 + u**2)) * u
    #h = lambda u: -((k**2 - sigma**2)**2 - sigma**4/4) * u / (1 + u**2)
    h = lambda u: -k**2*u
    #g = lambda u: sqrt(2)*u*sigma
    #g = lambda u: sqrt(2)*sigma*np.sqrt(1 + u**2)
    g = lambda u: sqrt(2)*sigma

    spde = SPDE(
        coeff,
        #lambda u, t: 0,
        lambda u, t: f(u) * dc1(u)**2 + h(u),
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
            verbose=False, 
            pbar=True, 
            seed=1,
            check=True,
    )

    ensemble_solver.solve()
    field = 0

    mean = ensemble_solver.means["value"][field]
    square = ensemble_solver.means["square"][field]

    step_errors = ensemble_solver.step_errors["value"]
    sample_errors = ensemble_solver.sample_errors["value"]

    sq_step_errors = ensemble_solver.step_errors["square"]
    sq_sample_errors = ensemble_solver.sample_errors["square"]

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

    fig2, ax2 = vis.steady_state(num_eq=equilibrate_over,
        label="Numerical solution", marker='.', linestyle='-.')

    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")

    t0 = lattice.range[0]
    ax2.plot(ts, np.exp(k*(ts - t0)), label="Analytical solution")


    fig, ax = vis2.surface(cstride=points//mesh_points,
                          rstride=int(steps/resolution)//mesh_points)

    fig3, ax3 = vis2.steady_state(num_eq=equilibrate_over,
        label="Numerical solution", marker='.', linestyle='-.')
    #ax3.plot(ts, (ts - t0)*(1-(ts - t0)))
    ax3.plot(ts, np.exp((2*k + sigma**2)*(ts - t0)), label="Geometric Brownian Motion")
    ax3.plot(ts, (boundaries[0]()**2 + sigma**2/(2*k)) * np.exp(2*k*(ts - t0)) - sigma**2/(2*k),
        label="Linear Diffusion")
    ax3.plot(ts, 
    (boundaries[0]()**2 + sigma**2/(2*k + sigma**2))*np.exp((2*k+sigma**2)*(ts - t0)) - sigma**2/(2*k + sigma**2),
        label="Multinoise")
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.legend()

    #taus = vis.taxis
    #fig, ax = vis.at_origin()
    #ax.plot(taus, np.exp(k*taus), linestyle='-.')

    np.save("lattice_points", lattice.points)
    np.save('mean', mean)
    np.save('square', square)
    np.save("mean_step_error", step_errors)
    np.save("mean_sample_error", sample_errors)
    np.save("square_step_error", sq_step_errors)
    np.save("square_sample_error", sq_sample_errors)


    plt.show()
