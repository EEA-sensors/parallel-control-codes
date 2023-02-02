"""
Model (in NumPy) which represents data and linear model for tracking a trajectory with a 2d constant velocity model.
That is, the model is

  dx1/dt = x3 + u1
  dx2/dt = x4 + u2

and the discrete-time cost function is

  1/2 sum_k (H x_k - r_k)^T X_k (H x_k - r_k)
  + 1/2 sum_k u_k^T U_k u_k

where H = [1 0 0 0; 0 1 0 0] and r_k forms a point trajectory. The continuous-time version is

  1/2 \int [(H x(t) - r(t))^T X (H x(t)) - r(t)) + \int u^T(t) U u(t),)

where r(t) is a curve fitted to the points r_k.


@author: Simo Särkkä
"""

import numpy as np
import math
import parallel_control.disc as disc
import parallel_control.lqt_np as lqt_np

###########################################################################

class LinearModel:

    def __init__(self, seed=123):
        self.seed = seed

    def genData(self, N):
        rng = np.random.default_rng(self.seed)

        x0 = 0.0
        y0 = 0.0
        x1 = 10.0
        y1 = 10.0
        r  = 1.0
        s  = r / 2.0
        da = 0.5 * np.pi

        xy = np.empty((2, N))
        x = x0 + (x1 - x0) / 2.0
        y = y0 + (y1 - y0) / 2.0
        xy[0, 0] = x
        xy[1, 0] = y

        new_x = x
        new_y = y
        angle = 0.0

        for k in range(1, N):
            angle = angle + da * 2.0 * (rng.uniform(0.0, 1.0) - 0.5)
            if angle < 0.0:
                angle = angle + 2.0 * np.pi
            if angle > 2.0 * np.pi:
                angle = angle - 2.0 * np.pi

            new_x = x + r * math.cos(angle)
            new_y = y + r * math.sin(angle)

            if new_x <= x0 + s or new_x >= x1 - s:
                new_x = 2.0 * x - new_x
            if new_y <= y0 + s or new_y >= y1 - s:
                new_y = 2.0 * y - new_y

            x = new_x
            y = new_y

            xy[0,k] = x
            xy[1,k] = y

        return xy

    def getLQT(self, xy, steps=10):
        dt = 1.0 / steps

        U = 0.1 * np.eye(2)
        H = np.array([[1.0,0.0,0.0,0.0],
                      [0.0,1.0,0.0,0.0]])
        HT = np.eye(4)
        Xl = 100.0  * np.eye(2)
        Xn = 1e-6 * np.eye(2)
        XT = 1.0  * np.eye(4)

        F = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
        Qc = np.diag(np.array([1,1]))

        L = np.array([[0.0,0.0],
                      [0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

        G = np.array([[0.0,0.0],
                      [0.0,0.0],
                      [1.0,0.0],
                      [0.0,1.0]])

        F, L, Q = disc.lti_disc_u(F, L, G, Qc, dt)

        N = xy.shape[1]
        T = steps * N

        i = 0
        curr_r = xy[:,0]
        x0 = np.array([curr_r[0],curr_r[1],0.0,0.0])
        X = []
        r = []
        for k in range(T):
            if k % steps == 0:
                X.append(Xl)
                curr_r = xy[:, i]
                i = i + 1
            else:
                X.append(Xn)
            r.append(curr_r)

        rT = np.array([curr_r[0],curr_r[1],0.0,0.0])

        lqt = lqt_np.LQT.checkAndExpand(F, L, X, U, XT, None, H, r, HT, rT, T=T)

        return lqt, x0

