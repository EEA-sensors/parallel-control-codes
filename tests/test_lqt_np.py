#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for numpy-based Linear Quadratic Regulator and Tracker routines.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.lqt_np as lqt_np
import unittest


##############################################################################
# Unit tests for LQR
##############################################################################

class LQR_np_UnitTest(unittest.TestCase):
    """Unit tests for LQR"""

    def setUp(self):
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])
        L = np.array([[0], [1]])

        X = np.diag([0.5, 0.1])
        U = np.array([[1]])
        XT = 0.1 * np.eye(2)

        self.lqr = lqt_np.LQR(F, L, X, U, XT)

    def test_nonstationary_lqr(self):
        S0 = np.array([[3.77592943, 0.81423906],
                       [0.81423906, 0.59689854]])
        Kx0 = np.array([[0.48664611, 0.41547464]])

        Kx_list, S_list = self.lqr.seqBackwardPass(10)

        self.assertTrue(linalg.norm(Kx_list[0] - Kx0) < 1e-6)
        self.assertTrue(linalg.norm(S_list[0] - S0) < 1e-6)

        u1 = np.array([-0.2805982])
        x1 = np.array([0.1, 0.58452536])

        x0 = np.array([0, 1])
        u_list, x_list = self.lqr.seqForwardPass(x0, Kx_list)

        self.assertTrue(linalg.norm(u_list[1] - u1) < 1e-6)
        self.assertTrue(linalg.norm(x_list[1] - x1) < 1e-6)

    def test_parallel_lqr(self):
        Kx_list1, S_list1 = self.lqr.seqBackwardPass(10)
        Kx_list2, S_list2 = self.lqr.parBackwardPass(10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-10)

        x0 = np.array([0, 1])
        u_list1, x_list1 = self.lqr.seqForwardPass(x0, Kx_list1)

        x0 = np.array([0, 1])
        u_list2, x_list2 = self.lqr.parForwardPass(x0, Kx_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_stationary_lqr(self):
        # This is from Matlab:
        # >> dt = 0.1; [K,S] = dlqr([1 dt; 0 1],[0;1],diag([0.5 0.1]),1)
        #
        # K =
        #   0.553250340445470   0.443153155638486
        #
        # S =
        #   4.004996682710235   0.903750008716494
        #   0.903750008716494   0.633528156510135

        K0 = np.array([0.553250340445470, 0.443153155638486])
        S0 = np.array([[4.004996682710235, 0.903750008716494],
                       [0.903750008716494, 0.633528156510135]])

        K, S = self.lqr.lqrDare()
        self.assertTrue(linalg.norm(K - K0) < 1e-10)
        self.assertTrue(linalg.norm(S - S0) < 1e-10)

        K, S = self.lqr.lqrDouble(10)
        self.assertTrue(linalg.norm(K - K0) < 1e-10)
        self.assertTrue(linalg.norm(S - S0) < 1e-10)

        K, S = self.lqr.lqrIter(100)
        self.assertTrue(linalg.norm(K - K0) < 1e-10)
        self.assertTrue(linalg.norm(S - S0) < 1e-10)

        Kx_list, S_list = self.lqr.seqBackwardPass(100)
        self.assertTrue(linalg.norm(Kx_list[0] - K0) < 1e-10)
        self.assertTrue(linalg.norm(S_list[0] - S0) < 1e-10)

    def test_single_step(self):
        Kx_list1, S_list1 = self.lqr.seqBackwardPass(1)
        Kx_list2, S_list2 = self.lqr.parBackwardPass(1)

        x0 = np.array([0, 1])
        u_list1, x_list1 = self.lqr.seqForwardPass(x0, Kx_list1)

        x0 = np.array([0, 1])
        u_list2, x_list2 = self.lqr.parForwardPass(x0, Kx_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)


##############################################################################
# Unit tests for LQT
##############################################################################

class LQT_np_UnitTest(unittest.TestCase):
    """Unit tests for LQT"""

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

    def test_against_seq_lqr(self):
        #
        # Test that sequential LQR and LQT give the same results when they should match
        #
        T = 10
        lqr, lqt = self.setupLQR(T)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()

        Kx_list2, S_list2 = lqr.seqBackwardPass(T)
        d_list2 = T * [np.array([0])]
        v_list2 = T * [np.array([0, 0])]

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-10)

        x0 = np.array([0, 1])
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        x0 = np.array([0, 1])
        u_list2, x_list2 = lqr.seqForwardPass(x0, Kx_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_againts_par_lqr(self):
        #
        # Test that parallel LQR and LQT give the same results when they should match
        #
        T = 10
        lqr, lqt = self.setupLQR(T)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.parBackwardPass()

        Kx_list2, S_list2 = lqr.parBackwardPass(T)
        d_list2 = T * [np.array([0])]
        v_list2 = T * [np.array([0, 0])]

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-10)

        x0 = np.array([0, 1])
        u_list1, x_list1 = lqt.parForwardPass(x0, Kx_list1, d_list1)

        x0 = np.array([0, 1])
        u_list2, x_list2 = lqr.parForwardPass(x0, Kx_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_lqr_batch(self):
        T = 5
        lqr, lqt = self.setupLQR(T)
        x0 = np.array([0.1, 1])

        u_list1, x_list1 = lqt.batchSolution(x0)

        Kx_list2, S_list2 = lqr.seqBackwardPass(T)
        u_list2, x_list2 = lqr.seqForwardPass(x0, Kx_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_seq_par_init_tracking(self):
        F = np.eye(4)
        L = np.eye(4,2)
        X = np.eye(3)
        U = np.eye(2)
        XT = np.eye(3)

        T = 10
        lqt = lqt_np.LQT.checkAndExpand(F, L, X, U, XT, T=T)

        x0 = np.ones(4)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        u_list2, x_list2 = lqt.parForwardPass(x0, Kx_list2, d_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_seq_par_single_step(self):
        F = np.eye(4)
        L = np.eye(4,2)
        X = np.eye(3)
        U = np.eye(2)
        XT = np.eye(3)

        T = 1
        lqt = lqt_np.LQT.checkAndExpand(F, L, X, U, XT, T=T)

        x0 = np.ones(4)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        u_list2, x_list2 = lqt.parForwardPass(x0, Kx_list2, d_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def setupRndLQT(self):
        #
        # Test by initializing with random matrices
        # while keeping the sizes the default
        #
        rng = np.random.default_rng(123)

        T = 5
        nx = 4
        nu = 2
        nr = 3

        x0 = rng.standard_normal(nx)

        # Randomize
        #        lqt.HT = rng.standard_normal((nx, nx))
        #        lqt.rT = rng.standard_normal(nx)
        #        lqt.XT = rng.standard_normal((nx, 2 * nx))
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

        lqt = lqt_np.LQT.checkAndExpand(F,L,X,U,XT,c,H,r,HT,rT,Z,s,M,T)

        return lqt, x0

    def test_seq_rnd_tracking_1(self):
        lqt, x0 = self.setupRndLQT()

        u_list0, x_list0 = lqt.batchSolution(x0)
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        x_list2 = lqt.seqSimulation(x0, u_list1)

        self.assertTrue(len(x_list0) == len(x_list1))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list0, x_list1)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(u_list0) == len(u_list1))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list0, u_list1)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(x_list2) == len(x_list1))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list2, x_list1)])
        self.assertTrue(err < 1e-10)

    def test_seq_par_rnd_tracking_1(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        u_list2, x_list2 = lqt.parForwardPass(x0, Kx_list2, d_list2)

        self.assertTrue(len(x_list1) == len(x_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(u_list1) == len(u_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_seq_par_rnd_tracking_2(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list2, x_list2 = lqt.parFwdBwdPass(x0, Kx_list2, d_list2, S_list2, v_list2)

        self.assertTrue(len(x_list1) == len(x_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(u_list1) == len(u_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_seq_par_rnd_tracking_3(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.parBackwardPass()
        u_list1, x_list1 = lqt.parFwdBwdPass2(x0, Kx_list1, d_list1, S_list1, v_list1)

        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list2, x_list2 = lqt.parFwdBwdPass(x0, Kx_list2, d_list2, S_list2, v_list2)

        self.assertTrue(len(x_list1) == len(x_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(u_list1) == len(u_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)

    def test_trans_pass(self):
        lqt, x0 = self.setupRndLQT()
        # This really works only for Zk = I
        for k in range(len(lqt.Z)):
            lqt.Z[k] = np.eye(lqt.Z[k].shape[0])

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.seqTransformedBackwardPass()

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-10)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-10)

    def test_cost(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        rng = np.random.default_rng(123)
        u_list2 = [u + rng.uniform(0, 0.01, size=u.shape) for u in u_list1]
        x_list2 = lqt.seqSimulation(x0, u_list2)

        cost1 = lqt.cost(x_list1, u_list1)
        cost2 = lqt.cost(x_list2, u_list2)

        #        print("cost1 = %f" % cost1)
        #        print("cost2 = %f" % cost2)
        self.assertTrue(cost2 > cost1)

