#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Numpy-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.lqt_np as lqt_np
import parallel_control.nlqt_np as nlqt_np
import unittest

class NLQT_np_testmodel:
    def f(self, x, u):
        return -np.sin(x) + u

    def Fx(self, x, u):
        return np.array([-np.cos(x)])

    def Fu(self, x, u):
        return np.array([[1.0]])

class NLQT_np_UnitTest(unittest.TestCase):
    """Unit tests for NLQT_np"""

    def setupNLQT(self):
        model = NLQT_np_testmodel()

        x = [np.array([x]) for x in [0.5,0.6,0.7,0.8,0.9,1.0]]
        u = [np.array([u]) for u in [0.1,0.2,0.1,0.2,0.1]]
        r = [np.array([x]) for x in [0.5,0.2,0.2,0.2,0.1]]
        F = np.array([[1]])
        L = np.array([[1]])
        X = np.array([[1]])
        U = np.array([[1]])
        XT = np.array([[1]])
        T = len(r)
        lqt = lqt_np.LQT.checkAndExpand(F,L,X,U,XT,r=r,T=T)

        nlqt = nlqt_np.NLQT(lqt, model)

        return nlqt, x, u, r

    def test_linearize(self):
        nlqt, x, u, r = self.setupNLQT()

        nlqt.linearize(u, x)

        F = nlqt.lqt.F
        c = nlqt.lqt.c
        L = nlqt.lqt.L

        self.assertTrue(np.linalg.norm(F[0] @ x[0] + c[0] + L[0] @ u[0] - nlqt.model.f(x[0],u[0])) < 1e-10)
        self.assertTrue(np.linalg.norm(F[1] @ x[1] + c[1] + L[1] @ u[1] - nlqt.model.f(x[1],u[1])) < 1e-10)

    def test_simulate(self):
        nlqt, x, u, r = self.setupNLQT()
        x2 = nlqt.simulate(x[0], u)
        self.assertTrue(np.abs(x2[1] - nlqt.model.f(x2[0],u[0])) < 1e-10)

    def test_cost(self):
        nlqt, x, u, r = self.setupNLQT()
        cost = nlqt.cost(x[0], u)
        print("cost = %f" % cost)

    def test_iterate(self):
        nlqt, x, u, r = self.setupNLQT()

        cost0 = nlqt.cost(x[0], u)
        cost = cost0
        for i in range(5):
            u, x = nlqt.iterate(u, x)
            cost = nlqt.cost(x[0], u)
            print("iter cost = %f" % cost)
        self.assertTrue(cost < cost0)

