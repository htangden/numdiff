from tqdm import tqdm
import numpy as np
from numpy.linalg import inv
import scipy as sp
from ivp import IVP

class Solver:
    def __init__(self, ivp : IVP, N : int, tol=1e-6, adaptive = False, k=0):
        self.ivp = ivp
        self.N = N
        self.tol = tol
        self.us = np.zeros((self.ivp.y0.shape[0], N+1))
        self.us[:,0] = self.ivp.y0
        self.k = k
        self.adaptive = adaptive
        self.label = "placeholder solver label"

        if not self.adaptive:
            self.grid = np.linspace(ivp.t0, ivp.T, N+1)
            self.h = (ivp.T - ivp.t0)/N
        else:
            self.us = np.array([self.ivp.y0]).T
            t0 = self.ivp.t0
            T = self.ivp.T
            self.grid = [t0]
            self.h = np.abs(T - t0) * tol**(1/self.k) / (100 * (1 + sp.linalg.norm(self.ivp.f(t0, self.us[:,0]))))

        self.errest = tol

    def setN(self, N):
        if self.adaptive:
            raise ValueError
        self.N = N
        self.grid = np.linspace(self.ivp.t0, self.ivp.T, N+1)
        self.h = (self.ivp.T - self.ivp.t0)/N
        self.us = np.zeros((self.ivp.y0.shape[0], N+1))
        self.us[:,0] = self.ivp.y0

    def step(self, i):
        raise NotImplementedError

    def newstep(self, rn):
        rn1 = self.errest
        self.h = min(
            ((self.tol / rn)**(2/(3*self.k)) * (self.tol / rn1)**(-1/(3*self.k)) * self.h),
            self.ivp.T - self.grid[-1]
        )
        self.errest = rn

    def integrate(self):
        if not self.adaptive:
            for i in tqdm(range(self.N), desc="Integrating (Fixed)"):
                    self.step(i)
        else:
            i = 0
            with tqdm(total=self.ivp.T, initial=self.grid[-1], desc="Integrating (Adaptive)") as pbar:
                while (self.grid[-1] < self.ivp.T):
                    self.step(i)
                    
                    pbar.n = int(self.grid[-1])
                    pbar.refresh()
                    i += 1

        return np.array(self.grid), self.us


class ExplicitEuler(Solver):
    def __init__(self, ivp : IVP, N : int):
        super().__init__(ivp, N)
        self.label = f"Explicit Euler: $N={N}$"

    def step(self, i):
        un = self.us[:,i]
        tn = self.grid[i]
        self.us[:,i+1] = un + self.h * self.ivp.f(tn, un)


class ImplicitEulerLTE(Solver):
    def __init__(self, ivp : IVP, N : int):
        super().__init__(ivp, N)
        self.label = f"Implicit Euler: $N={N}$"

    def step(self, i):
        un = self.us[:,i]
        self.us[:,i+1] = np.linalg.inv(np.eye(self.ivp.A.shape[0]) - self.h * self.ivp.A) @ un

class ImplicitEuler(Solver):
    def __init__(self, ivp: IVP, N: int):
        super().__init__(ivp, N)
        self.label = f"Implicit Euler: $N={N}$"

    def step(self, i):
        un = self.us[:,i]
        tn = self.grid[i]
        difference = lambda x: x - un - self.ivp.f(tn + self.h, x)
        self.us[:,i+1] = sp.optimize.root(difference, un, tol=1e-6)

class TrapezoidalLTE(Solver):
    def __init__(self, ivp : IVP, N : int):
        super().__init__(ivp, N)
        self.label = f"Trapezoidal Rule: $N={N}$"

    def step(self, i):
        A = self.ivp.A
        un = self.us[:,i]
        self.us[:,i+1] = np.linalg.inv(np.eye(A.shape[0]) - self.h/2 * A) @ (np.eye(A.shape[0]) + self.h/2 * A) @ un


class ExplicitLaTa(Solver):
    def __init__(self, ivp : IVP, N : int):
        super().__init__(ivp, N)
        self.label = f"Larsson-Tangden: $N={N}$"

    def step(self, i):
        u = self.us[:,i]
        t = self.grid[i]
        if (i>=1):
            u1 = self.us[:,i-1]
            t1 = self.grid[i-1]
            self.us[:,i+1] = u + self.h * (3/2 * self.ivp.f(t, u) - 1/2 * self.ivp.f(t1, u1))
        else:
            self.us[:,i+1] = u + self.h * self.ivp.f(t,u)

class RK4(Solver):
    def __init__(self, ivp : IVP, N : int):
        super().__init__(ivp, N)
        self.label = f"RK4: $N={N}$"
        
    def step(self, i):
        un = self.us[:,i]
        tn = self.grid[i]

        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h/2, un + self.h*Y1/2)
        Y3 = self.ivp.f(tn + self.h/2, un + self.h*Y2/2)
        Y4 = self.ivp.f(tn + self.h, un + self.h*Y3)

        self.us[:,i+1] = un + self.h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4)

class RK34(Solver):
    def __init__(self, ivp : IVP, N : int, tol : float = 1e-6):
        super().__init__(ivp, N, k=4, adaptive=True, tol=tol)
        self.label = f"RK34: $tol={tol:.2e}$"
        
    def step(self, i):
        un = self.us[:,i]
        self.grid.append(self.grid[-1] + self.h) #type: ignore
        tn = self.grid[i]

        Y1 = self.ivp.f(tn, un)
        Y2 = self.ivp.f(tn + self.h/2, un + self.h*Y1/2)
        Y3 = self.ivp.f(tn + self.h/2, un + self.h*Y2/2)
        Y4 = self.ivp.f(tn + self.h, un + self.h*Y3)

        Z3 = self.ivp.f(tn + self.h, un - self.h*(2*Y2 - Y1))

        error = sp.linalg.norm(self.h/6 * (2 * Y2 + Z3 - 2*Y3 - Y4))
        self.us = np.column_stack((self.us, (un + self.h/6 * (Y1 + 2*Y2 + 2*Y3 + Y4))))

        self.newstep(error)
