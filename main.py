from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver
from src.noises import WhiteNoise
from mpl_toolkits import mplot3d


if __name__ == "__main__":
    coeff = 1
    points = 10
    steps = 150**2
    tmax = 5
    samples = 100

    sigma = 1
    k = -1

    f = 1
    g = lambda u: k*u
    gderiv = lambda u: k
    #g = lambda u: -0.5*sigma*u*np.exp(-u**2)

    u0 = np.ones(points)
    noise = WhiteNoise(1, points)

    d1 = DerivativeOperator(1, 1/points, f, g)
    d2 = DerivativeOperator(2, 1/points, f, g)

    linear = lambda a, t, w: -k**2 * a + sigma*w
    geometric_brownian = lambda a, t, w: - d1(a)**2/a + a*sigma*w
    linmult_arnold = lambda a, t, w: -d1(a)**2 * a/(1 + a**2) - (k - sigma**2)**2 * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w
    linmult_graham = lambda a, t, w: -d1(a)**2 * a/(1 + a**2) - ((k - sigma**2)**2 - sigma**4/4) * a / (1 + a**2) + sigma * np.sqrt(1 + a**2)*w
    gaussian_arnold = lambda a, t, w: a*d1(a)**2 + sigma**4 / 4 * np.exp(-2*a**2) * (a**3 - 1.5*a) + sigma * np.exp(-a**2/2)*w

    spde = SPDE(coeff, linmult_arnold, noise, points, f, g, right_deriv=gderiv)

    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver)
    ensemble_solver = EnsembleSolver(solver, samples)
    ensemble_solver.solve()
    mean = ensemble_solver.mean
    square = ensemble_solver.square

    err = np.sqrt(square - mean**2)
    vis = Visualizer(mean, (0, tmax), (0, 1), error=ensemble_solver.sample_error)
    vis2 = Visualizer(square, (0, tmax), (0, 1), error=ensemble_solver.square_sample_error)

    fig, ax = vis.surface(cstride=10*points, rstride=10)
    ts = vis.xaxis

    fig2, ax2 = vis.steady_state(label="Numerical solution", marker='o', linestyle='-.')
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

    fig3, ax3 = vis2.steady_state('o', label="Numerical solution", marker='o', linestyle='-.')
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.plot(ts, 
        (f + sigma**2/(2*k+sigma**2))*np.exp((2*k + sigma**2)*ts) - sigma**2/(2*k + sigma**2), \
            label="Analytical, $B = \sigma\sqrt{1 + \phi^2}$")
    ax3.plot(xs, res.sol(xs)[0]**2, label="Scipy solution")
    ax3.legend()

    plt.show()