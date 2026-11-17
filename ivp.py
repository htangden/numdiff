import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from collections.abc import Callable

class IVP:
    def __init__(self, t0 : int, T : int, y0 : np.ndarray):
        self.t0 = t0
        self.T = T
        self.y0 = y0
        self.A = np.zeros((y0.shape[0], y0.shape[0]))

    def f(self, t, y):
        raise NotImplementedError

    def exact(self, t):
        raise NotImplementedError


class LTE(IVP):
    def __init__(self, t0, T, y0, A):
        super().__init__(t0, T, y0)
        self.A = A

    def f(self, t, y):
        return self.A @ y

    def exact(self, t):
        return sp.linalg.expm(t * self.A)@self.y0


class SquareIVP(IVP):
    def __init__(self, t0, T, y0):
        if np.any([t0 < 1/y0i < T for y0i in y0]):
            raise ValueError
        super().__init__(t0, T, y0)

    def f(self, t, y):
        return np.array([yi ** 2 for yi in y])

    def exact(self, t):
        return np.array([y0i / (1 - y0i*t) for y0i in self.y0])


class LotkaVolterra(IVP):
    def __init__(self, t0, T, y0, a, b, c, d):
        super().__init__(t0, T, y0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f(self, t, y):
        xp = self.a * y[0] - self.b * y[0] * y[1]
        yp = self.c * y[0] * y[1] - self.d * y[1]
        return np.array([xp, yp])

    def H(self, x, y):
        return self.c*x + self.b*y - self.d*np.log(x) - self.a*np.log(y)

class VanDerPol(IVP):
    def __init__(self, t0, T, y0, mu):
        super().__init__(t0, T, y0)
        self.mu = mu

    def f(self, t, y):
        y1p = y[1]
        y2p = self.mu * (1 - y[0]**2) * y[1] - y[0]
        return np.array([y1p, y2p])



   
