import numpy as np
import matplotlib.pyplot as plt
from ivp import *
from solver import *
from evaluator import *
from helper import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": (
        r"\usepackage{amsmath}"
    ),
})

def plotErrorExact(ivp : IVP, solvers : list[Solver], plot_error = True):
    plt.figure()
    plt.title(f"Approximate solution to {ivp.label}")

    ts = np.linspace(0, 1, 1000)
    ys = [ivp.exact(t) for t in ts]
    plt.plot(ts , ys, label="Exact solution")

    for solver in solvers:
        label = solver.label
        grid, us = solver.integrate()
        for i, row in enumerate(us):
            plt.scatter(grid, row, label=f"{label}")

        eval = Evaluator(ivp, solver)
        if plot_error:
            errs = eval.errvstime()
            plt.scatter(grid, errs, label=f"Error: {label}")

    plt.legend()

def plotLotka(ivp : IVP, solver : Solver):
    plt.figure()
    plt.title(f"Approximate solution to {ivp.label}")
    plt.ylabel("population")
    plt.xlabel("time")

    grid, us = solver.integrate()
    animals = ["Rabbit", "Fox"]
    for i, row in enumerate(us):
        plt.plot(grid, row, label=f"{solver.label} - {animals[i]}")

def plotVanDerPol(ivp : IVP, solver : Solver, label = "testing"):
    grid, us = solver.integrate()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel("$y_2$")
    ax1.plot(grid, us[1], label=f"$y_2$, {solver.label}", color="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Step size')
    ax2.set_yscale('log')
    ax2.plot(grid[:-1], np.diff(grid), label=f"Step size, {solver.label}", color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    plt.title(f"Approx. Solution (+ step size) to {ivp.label}")

def plotPhasePortrait(ivp : IVP, solver : Solver, label = "testing", axislabels = ["Rabbit", "Fox"]):
    plt.title(f"Phase portrait of {ivp.label}")
    _, us = solver.integrate()
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    plt.scatter(us[0], us[1], label=f"{label}")

def plotLotkaInvariance(ivp : LotkaVolterra, solver : Solver):
    plt.figure()
    plt.title(f"Invariance of {ivp.label}")
    plt.xlabel("Time")
    plt.ylabel(r"$\vert \frac{H}{H_0} - 1 \vert$")
    grid, us = solver.integrate()
    H0 = ivp.H(us[0,0], us[1,0])
    Hs = np.array([abs(ivp.H(us[0,i], us[1,i])/H0 - 1) for i in range(len(grid))])
    plt.semilogy(grid, Hs, label=solver.label)
    plt.legend()

def plotLogLog(ivp, solver, Ns = 2 ** np.arange(4, 10), label="placeholder"):
    eval = Evaluator(ivp, solver)
    hs, errs = eval.errvsh(Ns)
    slope = (np.log(errs[-1]) - np.log(errs[0]))/(np.log(hs[-1]) - np.log(hs[0]))
    plt.loglog(hs, errs, label=f"{label}, Order: ${slope:.2f}$")
    plt.xlabel("h")
    plt.ylabel("Error")

def plot_stiffness():
    rk34 = []
    rk45 = []
    bdf = []
    mus = [10, 15, 22, 33, 47, 68, 150, 220, 330, 470, 680, 1000]
    for i, mu in enumerate(mus):
            print(f"Working on mu = {mu}, {i+1}/{len(mus)}")
            t0 = 0
            T = 0.7*mu
            y0 = np.array([2, 0])
            ivp = VanDerPol(t0, T, y0, mu)
            solver = RK34(ivp)

            try:
                grid, _ = solver.integrate()
                rk34.append(len(grid)-1)
            except ValueError as e:
                rk34.append(0)
                print(e)

            try:
                sol_bdf = sp.integrate.solve_ivp(ivp.f, [t0, T], y0, atol=1e-6, rtol=1e-2, method="BDF")
                bdf.append(len(sol_bdf.t)-1)
            except ValueError as e:
                bdf.append(0)
                print(e)

            try:
                sol_rk45 = sp.integrate.solve_ivp(ivp.f, [t0, T], y0, atol=1e-6, rtol=1e-2, method="RK45")
                rk45.append(len(sol_rk45.t)-1)
            except ValueError as e:
                rk45.append(0)
                print(e)

        
    plt.figure()
    plt.title("Steps needed to approx. vdP depending on $\\mu$/stiffness")
    plt.xlabel("$\\mu$")
    plt.ylabel("Steps")
    plt.loglog(mus, rk34, label="Steps over mu, RK34")
    plt.loglog(mus, rk45, label="Steps over mu, RK45")
    plt.loglog(mus, bdf, label="Steps over mu, BDF")
    plt.loglog(mus, [mu**2 for mu in mus], label="Reference quadratic")
    plt.legend()

def plot_task22():
    t0 = 0
    T = 1
    N = 1000
    A = np.array([[3, 1], [5, 2]])
    y0 = np.array([0.7, 1.3])
    ivp = LTE(t0=t0, T=T, y0=y0, A=A)
    plt.figure()
    plt.title(r"Error over step size")

    solver1 = ExplicitEuler(ivp, N)
    plotLogLog(ivp, solver1, label=solver1.label)

    solver2 = RK4(ivp, N)
    plotLogLog(ivp, solver2, label=solver2.label)
    plt.legend()
    
    plt.figure()

def plot_task32():
    t0 = 0
    T = 15
    a = 3
    b = 9
    c = 15
    d = 15
    y0 = np.array([1.5 * d/c, 0.5*a/b])
    N = 1000
    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    solver = RK34(ivp, N)

    plotLotka(ivp, solver)
    plt.legend()

    plt.figure()
    for factor in [1.1, 1.5, 2.0]:
        y0 = np.array([factor * d/c, a/b/factor])
        ivp = LotkaVolterra(t0, T, y0, a, b, c, d)
        solver = RK34(ivp, N)
        plotPhasePortrait(ivp, solver, label=f"RK34, $y_0 = {array_to_latex(y0)}$")
    plt.legend()

    plt.figure()
    plotLotkaInvariance(ivp, solver)
    plt.legend()
 
def plot_task41():
    mu = 100
    t0 = 0
    T = 2 * mu
    y0 = np.array([2,0])
    ivp = VanDerPol(t0, T, y0, mu)
    solver = RK34(ivp)
    #plotVanDerPol(ivp, solver)
# , np.array([-1,2])
    y0s = [np.array([2, 50]), np.array([-2, 20])]
    plt.figure()
    axislabels=["$y_1$", "$y_2$"]
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    for y0 in y0s:
        ivp = VanDerPol(t0, T, y0, mu)
        solver = RK34(ivp)
        _, us = solver.integrate()
        plt.scatter(us[0], us[1], label=f"$y_0 = {array_to_latex(y0)}$", s=5, alpha=0.7)

    plt.legend()
    plt.title(f"Phase portrait different $y_0$ for van der Pol")


if __name__ == "__main__":
    t0 = 0
    T = 1
    A = np.array([[3, 1], [5, 2]])
    y0 = np.array([0.7, 1.3])

    ivp = LTE(t0=t0, T=T, y0=y0, A=A)

    solver1 = ExplicitEuler(ivp, N=50)
    solver2 = RK34(ivp)

    #plotErrorExact(ivp, [solver1, solver2])

    a = 3
    b = 9
    c = 15
    d = 15
    y0 = np.array([1.5 * d/c, 0.5*a/b])
    T = 150

    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)
    solver = RK34(ivp)

    #plotLotkaInvariance(ivp, solver)

    plt.show()



