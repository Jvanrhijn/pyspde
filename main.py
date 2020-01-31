from src.spde import *
from src.linear_solvers import GalerkinSolver
from src.noises import WhiteNoise
from mpl_toolkits import mplot3d


if __name__ == "__main__":
    coeff = 1
    points = 20
    steps = 100
    tmax = 5
    samples = 64

    sigma = sqrt(0.75)
    k = -1

    f = 1
    g = lambda u: (k - sigma**2)*u;
    #g = lambda u: -0.5*sigma*u*np.exp(-u**2)

    u0 = np.ones(points)
    noise = WhiteNoise(sqrt(2), points)

    d1 = DerivativeOperator(1, 1/points, f, g)

    geometric_brownian = lambda a, t, x, w: -d1(a)**2/a + a*sigma*w(t, x)
    linmult_arnold = lambda a, t, x, w: -d1(a)**2 * a/(1 + a**2) - (k - sigma**2)**2 * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w(t, x)
    linmult_graham = lambda a, t, x, w: -d1(a)**2 * a/(1 + a**2) - ((k - sigma**2)**2 - sigma**4/4) * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w(t, x)
    gaussian_arnold = lambda a, t, x, w: a*d1(a)**2 + sigma**4 / 4 * np.exp(-2*a**2) * (a**3 - 1.5*a) + sigma * np.exp(-a**2/2)*w(t, x)

    spde = SPDE(1, linmult_arnold, noise, points, f, g)
    #spde = SPDE(1, lambda a, t, x, w: sigma*w(t, x), noise, points, f, g)

    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver)
    ensemble_solver = EnsembleSolver(solver, samples)
    ensemble_solver.solve()
    mean = ensemble_solver.mean
    square = ensemble_solver.square

    vis = Visualizer(mean, (0, tmax), (0, 1), error=ensemble_solver.sample_error)
    vis2 = Visualizer(square, (0, tmax), (0, 1), error=ensemble_solver.square_sample_error)

    fig, ax = vis.surface()
    ts = vis.xaxis

    fig2, ax2 = vis.steady_state('o', label="Numerical solution")
    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")
    ax2.plot(ts, np.exp(k*ts), label="Analytical; $B = \sigma\sqrt{1 + \phi^2}$")

    """Now, let's compute the steady-state solution using Scipy"""
    from scipy.integrate import solve_bvp

    def du(t, u):                                                                      
        return np.vstack((u[1], u[0] / (1 + u[0]**2)*(u[1]**2 + (k - sigma**2)**2)))  
    
    def bc(ua, ub):
        return np.array([ua[0] - f, ub[1] - g(ub[0])])

    xs = np.linspace(0, 1, 100)
    ys = np.zeros((2, xs.size))
    res = solve_bvp(du, bc, xs, ys)
    ax2.plot(xs, res.sol(xs)[0], label="Scipy solution")
    """End Scipy solution""" 

    ax2.legend()

    fig3, ax3 = vis2.steady_state('o', label="Numerical solution")
    ax3.set_ylabel(r"$\langle\phi\rangle^2$")
    ax3.set_xlabel("t")
    ax3.plot(ts, 
        (f + sigma**2/(2*k+sigma**2))*np.exp((2*k + sigma**2)*ts) - sigma**2/(2*k + sigma**2), \
            label="Analytical, $B = \sigma\sqrt{1 + \phi^2}")
    ax3.plot(xs, res.sol(xs)[0]**2, label="Scipy solution")
    ax3.legend()

    plt.show()