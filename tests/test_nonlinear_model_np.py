#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy version of a nonlinear model for tracking position/orientation trajectory.

@author: Simo Särkkä
"""

import numpy as np
import math
import parallel_control.nonlinear_model_np as nonlinear_model_np

import unittest

class NonlinearModel_np_UnitTest(unittest.TestCase):
    """Unit tests for numpy nonlinear model"""

    def test_func(self):
        dt = 1.0
        q  = 1.0
        rob = nonlinear_model_np.NonlinearModel(dt, q)

        x = np.array([0.1,0.2,0.3,0.4])
        u = np.array([0.4,0.5])

        px = x[0] + x[3] * math.cos(x[2]) * dt
        py = x[1] + x[3] * math.sin(x[2]) * dt
        th = x[2] + u[1] * dt
        sp = x[3] + u[0] * dt

        self.assertTrue(np.linalg.norm(np.array([px,py,th,sp]) - rob.f(x, u)) < 1e-10)


    def test_der(self):
        dt = 1.0
        q  = 1.0
        rob = nonlinear_model_np.NonlinearModel(dt, q)

        x = np.array([0.1,0.2,0.3,0.4])
        u = np.array([0.4,0.5])

        h = 1e-10

        Fx = np.zeros((4,4))
        for i in range(Fx.shape[1]):
            dx = np.zeros(4)
            dx[i] = h
            Fx[:,i] = (rob.f(x + dx, u) - rob.f(x, u)) / h

        self.assertTrue(np.linalg.norm(Fx - rob.Fx(x,u)) < 1e-6)

        Fu = np.zeros((4,2))
        for i in range(Fu.shape[1]):
            du = np.zeros(2)
            du[i] = h
            Fu[:,i] = (rob.f(x, u + du) - rob.f(x, u)) / h

        self.assertTrue(np.linalg.norm(Fu - rob.Fu(x,u)) < 1e-6)

