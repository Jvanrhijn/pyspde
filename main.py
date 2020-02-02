from src.spde import *
from src.linear_solvers import GalerkinSolver, SpectralSolver
from src.noises import WhiteNoise
from mpl_toolkits import mplot3d
from examples.potentials import *


if __name__ == "__main__":

    coeff = 1
    points = 31
    steps = 150**2
    tmax = 5
    samples = 10
    processes = 4

    sigma = 0.1
    k = -1

    f = 1
    g = lambda u: (k - sigma**2)*u
    gderiv = lambda u: k - sigma**2
    #g = lambda u: -0.5*sigma*u*np.exp(-u**2)

    u0 = np.ones(points)
    noise = WhiteNoise(2, points)

    d1 = DerivativeOperator(1, 1/points, f, g)
    d2 = DerivativeOperator(2, 1/points, f, g)

    spde = SPDE(coeff, linmult_arnold(points, k, sigma, f, g), noise, points, f, g, right_deriv=gderiv)

    solver = TrajectorySolver(spde, steps, tmax, u0, GalerkinSolver)
    ensemble_solver = EnsembleSolver(solver, samples, processes=processes)
    ensemble_solver.solve()
    mean = ensemble_solver.mean
    square = ensemble_solver.square

    vis = Visualizer(mean, (0, tmax), (0, 1), 
        sample_error=ensemble_solver.sample_error,
        step_error=ensemble_solver.step_error)
    vis2 = Visualizer(square, (0, tmax), (0, 1), 
        sample_error=ensemble_solver.square_sample_error,
        step_error=ensemble_solver.square_step_error)

    mesh_points = min([20, points, steps])
    fig, ax = vis.surface(cstride=points//mesh_points, rstride=steps//mesh_points)
    ts = vis.xaxis

    fig2, ax2 = vis.steady_state(label="Numerical solution", marker='o', linestyle='-.')
    ax2.set_ylabel(r"$\langle\phi\rangle$")
    ax2.set_xlabel("t")
    ax2.plot(ts, np.exp(k*ts), label="Analytical; $B = \sigma\sqrt{1 + \phi^2}$")

    ax2.legend()

    fig3, ax3 = vis2.steady_state('o', label="Numerical solution", marker='o', linestyle='-.')
    ax3.set_ylabel(r"$\langle\phi^2\rangle$")
    ax3.set_xlabel("t")
    ax3.plot(ts, 
        (f + sigma**2/(2*k+sigma**2))*np.exp((2*k + sigma**2)*ts) - sigma**2/(2*k + sigma**2), \
            label="Analytical, $B = \sigma\sqrt{1 + \phi^2}$")

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
    ax3.plot(xs, res.sol(xs)[0]**2, label="Scipy solution")
    """End Scipy solution""" 


    ax3.legend()

    plt.show()