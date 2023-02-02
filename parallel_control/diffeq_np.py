"""
Numpy routines for ordinary and partial differential equations

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np

def rk4(f, dt, x, t=None, param=None):
    if t is None:
        if param is None:
            dx1 = f(x) * dt
            dx2 = f(x + 0.5 * dx1) * dt
            dx3 = f(x + 0.5 * dx2) * dt
            dx4 = f(x + dx3) * dt
        else:
            dx1 = f(x, param) * dt
            dx2 = f(x + 0.5 * dx1, param) * dt
            dx3 = f(x + 0.5 * dx2, param) * dt
            dx4 = f(x + dx3, param) * dt
    else:
        if param is None:
            dx1 = f(x, t) * dt
            dx2 = f(x + 0.5 * dx1, t + dt/2) * dt
            dx3 = f(x + 0.5 * dx2, t + dt/2) * dt
            dx4 = f(x + dx3, t + dt) * dt
        else:
            dx1 = f(x, t, param) * dt
            dx2 = f(x + 0.5 * dx1, t + dt/2, param) * dt
            dx3 = f(x + 0.5 * dx2, t + dt/2, param) * dt
            dx4 = f(x + dx3, t + dt, param) * dt

    x = x + (1.0 / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)

    return x
