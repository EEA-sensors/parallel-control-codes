"""
Model which represents linear model for controlling a chain of N masses m connected with springs
with constant c and dampers with constants d. The control is applied to the first and last mass.
The model is

  d2y_1/dt2 = c/m [-2 y_1 + y_2] + d/m [-2 dy_1/dt + dy_2/dt] + 1/m u_1
  d2y_i/dt2 = c/m [y_{i-1} - 2 y_i + y_{i+1}] + d/m [dy_{i+1}/dt - 2 dy_i/dt + dy_{i+1}/dt], i=2,...,N-1
  d2y_N/dt2 = c/m [-2 y_N + y_{N-1}] + d/m [-2 dy_N/dt + dy_{N-1}}/dt] - 1/m u_2

and the cost function is

  1/2 sum_k x_k^T X x_k + 1/2 sum_k u_k^T U u_k + 1/2 x_T^T X x_T

@author: Simo Särkkä
"""

import numpy as np
import math
import parallel_control.disc as disc
import parallel_control.lqt_np as lqt_np

###########################################################################

class MassModel:

    def __init__(self, N=5):
        self.X  = np.eye(2*N)
        self.U  = 0.1 * np.eye(2)
        self.XT = np.eye(2*N)

        m = 1.0
        c = 1.0
        d = 0.2

        self.Fc = np.zeros((2*N,2*N))
        for i in range(N):
            self.Fc[2*i,2*i+1] = 1/m
            self.Fc[2 * i + 1, 2 * i]     = -2 * c / m
            self.Fc[2 * i + 1, 2 * i + 1] = -2 * d / m
            if (i > 0):
                self.Fc[2*i+1,2*(i-1)]   = c/m
                self.Fc[2*i+1,2*(i-1)+1] = d/m
            if (i < N-1):
                self.Fc[2*i+1,2*(i+1)]   = c/m
                self.Fc[2*i+1,2*(i+1)+1] = d/m

        self.Lc = np.zeros((2*N,2))
        self.Lc[1,0]  = 1.0/m
        self.Lc[-1,1] = -1.0/m

        self.x0 = np.zeros(2*N)
        self.x0[0] = 1
        if (N % 2) == 0:
            self.x0[N] = 1
        else:
            self.x0[N-1] = 1

    def getLQT(self, dt=0.005, Tf=10.0):
        G  = self.Lc
        Qc = np.eye(self.Lc.shape[1])
        F, L, Q = disc.lti_disc_u(self.Fc, self.Lc, G, Qc, dt)

        T = int(Tf/dt)

        lqt = lqt_np.LQT.checkAndExpand(F, L, self.X, self.U, self.XT, T=T)

        return lqt, self.x0
