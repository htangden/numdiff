import numpy as np
import matplotlib.pyplot as plt
from ivp import *
from solver import *
from evaluator import *

def plotErrorExact(ivp, solver : Solver, label = "testing", plot_exact = True, plot_error = True):
    if plot_exact:
        ts = np.linspace(0, 1, 1000)
        ys = [ivp.exact(t) for t in ts]
        plt.plot(ts , ys, label="Exact solution")

    grid, us = solver.integrate()
    for i, row in enumerate(us):
        plt.scatter(grid, row, label=f"{label}")

    eval = Evaluator(ivp, solver)
    if plot_error:
        errs = eval.errvstime()
        plt.scatter(grid, errs, label=f"Error: {label}")

    # lata = ImplicitEulerLTE(ivp=ivp, N = N)
    # grid, us = lata.integrate()
    # for row in us:
    #     plt.scatter(grid, row)

    # eval = Evaluator(ivp, lata)
    # errs = eval.errvstime()
    # plt.scatter(grid, errs)

def plotLotka(ivp : IVP, solver : Solver, label = "testing", plot_exact = True, plot_error = True):
    if plot_exact:
        ts = np.linspace(0, 1, 1000)
        ys = [ivp.exact(t) for t in ts]
        plt.plot(ts , ys, label="Exact solution")

    grid, us = solver.integrate()
    animals = ["Rabbit", "Fox"]
    for i, row in enumerate(us):
        plt.scatter(grid, row, label=f"{label} - {animals[i]}")

    eval = Evaluator(ivp, solver)
    if plot_error:
        errs = eval.errvstime()
        plt.scatter(grid, errs, label=f"Error: {label}")

def plotVanDerPol(ivp : IVP, solver : Solver, label = "testing"):
    plt.figure()
    grid, us = solver.integrate()
    plt.plot(grid[:-1], np.log(np.diff(grid)), label="Step size")
    plt.plot(grid, us[1], label=f"{label}")


def plotPhasePortrait(ivp : IVP, solver : Solver, label = "testing"):
    _, us = solver.integrate()
    animals = ["Rabbit", "Fox"]
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
    plt.loglog(hs, errs, label=f"{label}, Order: {slope:.2f}")
    plt.xlabel("h")
    plt.ylabel("error")

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


if __name__ == "__main__":
    #plot_stiffness()
    #A = np.array([[3, 1], [5, 2]])
        #y0 = np.array([0.7, 1.3])
    #ivp = LTE(t0=t0, T=T, y0=y0, A=A)
    #ivp = SquareIVP(t0=t0, T=T, y0=y0)
#     a = 3
#     b = 9
#     c = 15
#     d = 15
#     y0 = np.array([1.5 * d/c, 0.5*a/b])

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
    mu = 10000
    t0 = 0
    T = 0.7*mu
    y0 = np.array([2, 0])
    ivp = VanDerPol(t0, T, y0, mu)
    sol_bdf = sp.integrate.solve_ivp(ivp.f, [t0, T], y0, atol=1e-6, method="BDF")
    plt.plot(sol_bdf.t, sol_bdf.y[1])

    plt.show()

    #plt.legend(["2euler", "2trap"," 3euler", "3trap"])


