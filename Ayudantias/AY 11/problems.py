import numpy as np

class ScalarWaveProblem:

    def __init__(self, t0tf, u0, v0, f=None, c=1, c0=1.0, coords=(0, 1, 0, 1)):
        self.coords = coords
        self.t0tf = t0tf
        self.u0 = u0
        self.v0 = v0
        if f is None:
            self.f = lambda t, x, y: x*0
        else:
            self.f = f
        if (type(c) == int) or (type(c) == np.float64):
            self.c = lambda t, x, y: c
        else:
            self.c = c
        if type(c0) != float:
            raise ValueError("c0 must be float")
        self.c0 = c0

class PoissonProblem:

    def __init__(self, f=None, k=0, coords=(0, 1, 0, 1)):
        self.coords = coords
        if f is None:
            self.f = lambda t, x, y: x*0
        else:
            self.f = f
        self.k = k
