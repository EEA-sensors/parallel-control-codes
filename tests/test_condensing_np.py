#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for numpy version of partial condensing for LQT.

@author: Simo Särkkä
"""

import unittest
from scipy import linalg
import numpy as np
import pprint
import parallel_control.lqt_np as lqt_np
import parallel_control.condensing_np as condensing_np

class LQTCondenser_np_UnitTest(unittest.TestCase):
    def setupLQR(self, T):
        #
        # Set up an LQR model for test cases
        #
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])
        L = np.array([[dt ** 2 / 2], [dt]])

        X = np.diag([0.5, 0.1])
        U = np.array([[1]])
        XT = 0.1 * np.eye(2)

        lqr = lqt_np.LQR(F, L, X, U, XT)
        lqt = lqt_np.LQT.checkAndExpand(F, L, X, U, XT, T=T)

        return lqr, lqt

    def test_condensing_1(self):
        T = 6
        lqr, lqt = self.setupLQR(T)

        cond = condensing_np.LQTCondenser()
        Nc = 3
        clqt = cond.condense(lqt, Nc)

        x0 = np.ones(2)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        print(u_list1)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2, x_list2 = clqt.seqForwardPass(x0, Kx_list2, d_list2)

        print(u_list2)

        err = linalg.norm(np.concatenate(u_list1) - np.concatenate(u_list2))
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1[::Nc], x_list2)])
        self.assertTrue(err < 1e-10)

    def test_condensing_2(self):
        T = 6
        lqr, lqt = self.setupLQR(T)

        cond = condensing_np.LQTCondenser()
        Nc = 4
        clqt = cond.condense(lqt, Nc)

        x0 = np.ones(2)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        print(u_list1)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2, x_list2 = clqt.seqForwardPass(x0, Kx_list2, d_list2)

        print(u_list2)

        cu_list1 = np.concatenate(u_list1)
        cu_list2 = np.concatenate(u_list2)
        cu_list2 = cu_list2[:cu_list1.shape[0]]

        err = linalg.norm(cu_list1 - cu_list2)
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1[:-1:Nc], x_list2[:-1])])
        self.assertTrue(err < 1e-10)

        err = linalg.norm(x_list1[-1] - x_list2[-1])
        self.assertTrue(err < 1e-10)

    def test_convertXY_1(self):
        T = 9
        lqr, lqt = self.setupLQR(T)

        cond = condensing_np.LQTCondenser()
        Nc = 4
        clqt = cond.condense(lqt, Nc)

        x0 = np.ones(2)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2)
        u_list2, x_list2 = cond.convertUX(u_list2a, x_list2a)

        print(u_list1)
        print(u_list2)
        print(u_list2a)
        print(x_list1)
        print(x_list2)
        print(x_list2a)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

    def setupRndLQT(self, T=11):
        #
        # Test by initializing with random matrices
        # while keeping the sizes the default
        #
        rng = np.random.default_rng(123)

        nx = 4
        nu = 2
        nr = 3

        x0 = rng.standard_normal(nx)

        # Randomize
        c = T * [0]
        F = T * [0]
        s = T * [0]
        r = T * [0]
        M = T * [0]
        Z = T * [0]
        H = T * [0]
        L = T * [0]
        X = T * [0]
        U = T * [0]

        HT = rng.standard_normal((nr, nx))
        rT = rng.standard_normal(nr)
        XT = rng.standard_normal((nr, 2 * nr))
        XT = XT @ XT.T
        for i in range(T):
            c[i] = rng.standard_normal(nx)
            F[i] = rng.standard_normal((nx, nx))
            s[i] = rng.standard_normal(nu)
            r[i] = rng.standard_normal(nr)
            M[i] = rng.standard_normal((nr, nu))
            Z[i] = rng.standard_normal((nu, nu))
            H[i] = rng.standard_normal((nr, nx))
            L[i] = rng.standard_normal((nx, nu))
            X[i] = rng.standard_normal((nr, 2 * nr))
            X[i] = X[i] @ X[i].T
            U[i] = rng.standard_normal((nu, 2 * nu))
            U[i] = U[i] @ U[i].T + np.eye(nu)

        lqt = lqt_np.LQT.checkAndExpand(F,L,X,U,XT,c,H,r,HT,rT,None,None,None,T)

        return lqt, x0

    def test_convertXY_2(self):

        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        cond = condensing_np.LQTCondenser()
        Nc = 3
        clqt = cond.condense(lqt, Nc)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2)
        u_list2, x_list2 = cond.convertUX(u_list2a, x_list2a)

        print(u_list1)
        print(u_list2)
        print(x_list1)
        print(x_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-8)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-8)

    def test_single_block_1(self):
        lqt, x0 = self.setupRndLQT(3)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        cond = condensing_np.LQTCondenser()
        Nc = 3
        clqt = cond.condense(lqt, Nc)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2)
        u_list2, x_list2 = cond.convertUX(u_list2a, x_list2a)

        print("Single block 1:")
        print(clqt.F)
        print(x_list2a)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-8)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-8)

    def test_single_block_2(self):
        lqt, x0 = self.setupRndLQT(3)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        cond = condensing_np.LQTCondenser()
        Nc = 7
        clqt = cond.condense(lqt, Nc)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass()
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2)
        u_list2, x_list2 = cond.convertUX(u_list2a, x_list2a)

        print("Single block 2:")
        print(clqt.F)
        print(x_list2a)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-8)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-8)
