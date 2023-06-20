"""
Numpy version of a nonlinear model for tracking position/orientation trajectory.

@author: Simo Särkkä
"""

import numpy as np
import math
import parallel_control.lqt_np as lqt_np

import unittest

###########################################################################
#
# Nonlinear model class
#
###########################################################################

class NonlinearModel:
    def __init__(self, dt=0.1, seed=123):
        """ Constructor.

        Parameters:
            dt: Sampling period.
            seed: Random seed.
        """
        self.dt = dt
        self.seed = seed

    def f(self, x, u):
        """ Nonlinear model function.

        Parameters:
            x: State vector.
            u: Control vector.

        Returns:
            x_new: New state vector.
        """
        # x = [px,py,theta,sp], u = [acc,turnrate]
        x_new = np.array([x[0] + x[3] * math.cos(x[2]) * self.dt,
                          x[1] + x[3] * math.sin(x[2]) * self.dt,
                          x[2] + u[1] * self.dt,
                          x[3] + u[0] * self.dt])

        return x_new

    def Fx(self, x, u):
        """ Jacobian of f w.r.t. x.

        Parameters:
            x: State vector.
            u: Control vector.

        Returns:
            Fx: Jacobian of f w.r.t. x.
        """
        d_dx = np.array([[1.0,0.0,-x[3] * math.sin(x[2]) * self.dt,math.cos(x[2]) * self.dt],
                         [0.0,1.0, x[3] * math.cos(x[2]) * self.dt,math.sin(x[2]) * self.dt],
                         [0.0,0.0,1.0,0.0],
                         [0.0,0.0,0.0,1.0]])

        return d_dx

    def Fu(self, x, u):
        """ Jacobian of f w.r.t. u.

        Parameters:
            x: State vector.
            u: Control vector.

        Returns:
            Fu: Jacobian of f w.r.t. u.
        """
        d_du = np.array([[0.0,0.0],
                         [0.0,0.0],
                         [0.0,self.dt],
                         [self.dt,0.0]])

        return d_du

    def genData(self, N, steps=10):
        """ Generate data from the model.

        Parameters:
            N: Total number of data points.
            steps: Stepping of the returned measurements.

        Returns:
            xyt: Numpy array of [x,y,theta] data.
            xyt_dense: Numpy array of [x,y,theta] data with steps=1.

        """
        tt = np.linspace(0, 2 * np.pi, N)
        x_data = 7 + 4.0 * np.cos(tt) + 2.0 * np.sin(2 * tt) + 2.0 * np.cos(3 * tt)
        y_data = 7 + 3.0 * np.sin(tt) - 1.0 * np.sin(2 * tt) + 2.0 * np.sin(3 * tt)
        t_data = np.zeros_like(x_data)
        for k in range(len(t_data)):
            if k < len(t_data) - 1:
                t_data[k] = np.arctan2(y_data[k + 1] - y_data[k], x_data[k + 1] - x_data[k])
            else:
                t_data[k] = t_data[k - 1]
        t_data = np.unwrap(t_data)

        rx_data = x_data[0::steps]
        ry_data = y_data[0::steps]
        rt_data = t_data[0::steps]

        xyt = np.stack([rx_data, ry_data, rt_data])
        xyt_dense = np.stack([x_data, y_data, t_data])

        return xyt, xyt_dense

    def initialGuess(self, lqt, x0, init_to_zero=True):
        """" Generate initial guess for the nonlinear LQT.

        Parameters:
            lqt: LQT object.
            x0: Initial state.
            init_to_zero: If True, initial guess is set to zero.

        Returns:
            x_list: List of state vectors.
            u_list: List of control vectors.
        """
        x_list = [x0]
        u_list = []

        if init_to_zero:
            for k in range(len(lqt.F)):
                x = np.zeros(lqt.F[k].shape[0])
                u = np.zeros(lqt.L[k].shape[1])
                x_list.append(x)
                u_list.append(u)
        else:
            curr_x = lqt.r[0][0]
            curr_y = lqt.r[0][1]
            curr_s = 0.0
            count = 0
            for k in range(len(lqt.F)):
                x = np.zeros(lqt.F[k].shape[0])
                x[0] = lqt.r[k][0]
                x[1] = lqt.r[k][1]
                x[2] = lqt.r[k][2]
                count = count + 1
                if np.abs(x[0] - curr_x) > 0.1 and np.abs(x[1] - curr_y) > 0.1:
                    curr_s = math.sqrt((x[0] - curr_x) ** 2 + (x[1] - curr_y) ** 2) / (count * self.dt)
                    curr_x = x[0]
                    curr_y = x[1]
                    count = 0
                x[3] = curr_s
                u = np.zeros(lqt.L[k].shape[1])
                x_list.append(x)
                u_list.append(u)

        return u_list, x_list

    def getLQT(self, xyt, steps=10):
        """ Get the LQT for the model.

        Parameters:
            xyt: Numpy array of [x,y,theta] data.
            steps: Stepping of the returned measurements.

        Returns:
            lqt: LQT object.
            x0: Initial state.
        """
        U = np.diag([1.0,100.0])
        H = np.array([[1.0,0.0,0.0,0.0],
                      [0.0,1.0,0.0,0.0],
                      [0.0,0.0,1.0,0.0]])
        HT = H
        Xl = np.diag([100.0,100.0,1000.0])
        Xn = 1e-6 * np.eye(3)
        XT = Xn

        F = np.eye(4)
        L = np.array([[0.0,0.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        N = xyt.shape[1]
        T = steps * N

        i = 0
        curr_r = xyt[:,0]
        x0 = np.array([curr_r[0],curr_r[1],curr_r[2],0.0])
        X = []
        r = []
        for k in range(steps * N):
            if k % steps == 0:
                X.append(Xl)
                curr_r = xyt[:, i]
                i = i + 1
            else:
                X.append(Xn)
            r.append(curr_r)

        rT = np.array([curr_r[0],curr_r[1],curr_r[2]])

        lqt = lqt_np.LQT.checkAndExpand(F, L, X, U, XT, None, H, r, HT, rT, T=T)

        return lqt, x0


