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
    plt.figure()
    grid, us = solver.integrate()
    plt.plot(grid[:-1], np.log(np.diff(grid)), label="Step size")
    plt.plot(grid, us[1], label=f"{label}")

def plotPhasePortrait(ivp : IVP, solver : Solver, label = "testing", axislabels = ["Rabbit", "Fox"]):
    plt.title("Phase portrait")
    _, us = solver.integrate()
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    plt.scatter(us[0], us[1], label=f"{label}")

def plotLotkaInvariance(ivp : LotkaVolterra, solver : Solver):
    grid, us = solver.integrate()
    H0 = ivp.H(us[0,0], us[1,0])
    Hs = np.array([abs(ivp.H(us[0,i], us[1,i])/H0 - 1) for i in range(len(grid))])
    plt.semilogy(grid, Hs)

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
            solver = RK34(ivp, N=100)

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
    plotLogLog(ivp, solver1, label="Explicit Euler")

    solver2 = RK4(ivp, N)
    plotLogLog(ivp, solver2, label="RK4")
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
 

if __name__ == "__main__":
    t0 = 0
    T = 1
    A = np.array([[3, 1], [5, 2]])
    y0 = np.array([0.7, 1.3])

    ivp = LTE(t0=t0, T=T, y0=y0, A=A)

    solver1 = ExplicitEuler(ivp, N=50)
    solver2 = RK4(ivp, N=50)

    plotErrorExact(ivp, [solver1, solver2])

    a = 3
    b = 9
    c = 15
    d = 15
    y0 = np.array([1.5 * d/c, 0.5*a/b])

#    ivp = LotkaVolterra(t0, T, y0, a, b, c, d)

    #solver1 = ExplicitEuler(ivp, N)
    #plotLogLog(ivp, solver1, label="Euler")

    #solver2 = RK34(ivp, N)
    #plotLogLog(ivp, solver2, label="RK4")

    #plot_error = False
    #plotLotka(ivp, solver1, label="Explicit Euler", plot_exact=False, plot_error=plot_error)
   # plotVanDerPol(ivp, solver2, label="RK34")
   # plt.legend()

   # plt.figure()
   # plotPhasePortrait(ivp, solver2, label="RK34")
   # plt.legend()

    #plt.figure()
    #plotLotkaInvariance(ivp, solver2)
    # plt.figure()
    # mu = 10000
    # t0 = 0
    # T = 0.7*mu
    # y0 = np.array([2, 0])
    # ivp = VanDerPol(t0, T, y0, mu)
    # sol_bdf = sp.integrate.solve_ivp(ivp.f, [t0, T], y0, atol=1e-6, method="BDF")
    # plt.plot(sol_bdf.t, sol_bdf.y[1])

    plt.show()



