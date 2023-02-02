#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for numpy differential equation routines.

@author: Simo Särkkä
"""

import unittest
import numpy as np
from scipy import linalg
import parallel_control.diffeq_np as diffeq_np

class DiffEq_np_UnitTest(unittest.TestCase):
    def test_rk4_1(self):

        # dx/dt = -c x, x(0) = x0
        #  x(t) = x0 exp(-c t)
        c  = 0.5
        x0 = 0.7
        f  = lambda x: -c * x

        dt = 0.1
        x1 = diffeq_np.rk4(f, dt, x0)
        x2 = x0 * np.exp(-c * dt)
        self.assertTrue(np.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, dt, x1)
        x2 = x0 * np.exp(-c * 1.0)

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

    def test_rk4_2(self):

        # dx/dt = -cos(t) x, x(0) = x0
        #  x(t) = x0 exp(-int_0^t cos(t) dt) = x0 exp(-sin(t))

        x0 = 0.7
        f = lambda x, t: -np.cos(t) * x

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, dt, x1, k * dt)
        x2 = x0 * np.exp(-np.sin(1.0))

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

    def test_rk4_3(self):
        x0 = np.array([1,2,3])
        F  = np.array([[0,1,0],[0,0,1],[-0.1,-0.2,-0.3]])

        f = lambda x: F @ x
        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, dt, x1)
        x2 = linalg.expm(F * 1.0) @ x0

        self.assertTrue(linalg.norm(x2 - x1) < 1e-5)

    def test_rk4_4(self):
        # dx/dt = -c x, x(T) = x0
        #  x(t) = x0 exp(-c t + c T)
        c  = 0.5
        x0 = 0.7
        f  = lambda x: -c * x

        dt = 0.1
        x1 = diffeq_np.rk4(f, -dt, x0)
        x2 = x0 * np.exp(-c * 0 + c * dt)

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, -dt, x1)
        x2 = x0 * np.exp(-c * 0 + c * 1.0)

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

    def test_rk4_5(self):

        # dx/dt = -c x, x(0) = x0
        #  x(t) = x0 exp(-c t)
        f  = lambda x, c: -c * x
        c  = 0.5
        x0 = 0.7

        dt = 0.1
        x1 = diffeq_np.rk4(f, dt, x0, param=c)
        x2 = x0 * np.exp(-c * dt)
        self.assertTrue(np.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, dt, x1, param=c)
        x2 = x0 * np.exp(-c * 1.0)

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

    def test_rk4_6(self):

        # dx/dt = -c cos(t) x, x(0) = x0
        #  x(t) = x0 exp(-int_0^t c cos(t) dt) = x0 exp(-c sin(t))

        x0 = 0.7
        f = lambda x, t, c: -c * np.cos(t) * x
        c = 2

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_np.rk4(f, dt, x1, k * dt, c)
        x2 = x0 * np.exp(-c*np.sin(1.0))

        self.assertTrue(np.abs(x2 - x1) < 1e-5)

