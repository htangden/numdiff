import numpy as np
import scipy as sp
from ivp import IVP
from solver import Solver

class Evaluator:
    def __init__(self, ivp : IVP, sol : Solver):
        self.ivp = ivp
        self.solver = sol

    def errvstime(self):
        _, us = self.solver.integrate()
        ys = np.array([self.ivp.exact(t) for t in self.solver.grid]).T
        N = len(self.solver.grid) - 1
        errs = np.arange(N+1, dtype=np.float64)
        for i in range(N+1):
            errs[i] = sp.linalg.norm(us[:,i] - ys[:,i])
        return errs
    
    def errvsh(self, Ns : np.ndarray):
        hs = np.zeros_like(Ns, dtype=np.float64)
        errs = np.zeros_like(Ns, dtype=np.float64)

        for i, N in enumerate(Ns):
            self.solver.setN(N)
            grid, usEstimation = self.solver.integrate()
            y = np.array([self.ivp.exact(self.ivp.T)])
            hs[i] = self.solver.h
            errs[i] = sp.linalg.norm(usEstimation[:,N] - y)

        return hs, errs




