#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Parareal CLQT.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.lqt_np as lqt_np
import parallel_control.clqt_np as clqt_np
import parallel_control.parareal_clqt_np as para_np
import math
import unittest

import matplotlib.pyplot as plt

##############################################################################
# Unit tests for Parareal CLQR
##############################################################################

class Parareal_CLQR_np_UnitTest(unittest.TestCase):
    """Unit tests for Parareal CLQR"""

    def getSimpleCLQR(self,T=1.0):
        F = np.array([[0,1,0],[0,0,1],[0,0,-0.1]])
        L = np.array([[0,0],[1,0],[0,1]])
        X = 0.01 * np.eye(3)
        U = 0.02 * np.eye(2)
        XT = X
        return clqt_np.CLQR(F, L, X, U, XT, T)

    def getSimpleCLQT(self,T=1.0):
        clqr = self.getSimpleCLQR(T)
        c = np.zeros((3,))
        H = np.eye(3)
        r = np.zeros((3,))
        HT = np.eye(3)
        rT = np.zeros((3,))
        clqt = clqt_np.CLQT(lambda t: clqr.F, lambda t: clqr.L, lambda t: clqr.X, lambda t: clqr.U, clqr.XT,
                            lambda t: c, lambda t: H, lambda t: r, HT, rT, T)
        return clqt, clqr

    def getRandomCLQT(self,T=10.0):
        X = lambda t: (0.02 + 0.01 * np.sin(1.0 * t)) * np.eye(2)
        U = lambda t: (0.04 + 0.01 * np.cos(0.3 * t)) * np.eye(1)
        XT = 0.02 * np.eye(2)
        F = lambda t: np.array([[0, 1], [-(2.0 * np.sin(3.1 * t)) ** 2, 0]])
        L = lambda t: np.array([[0], [1 + 0.1 * np.sin(2.1 * t)]])
        HT = np.array([[1.0,0.1],[0.0,1.0]])
        H = lambda t: np.array([[1.0,0.0],[0.1,1.0]])
        c = lambda t: np.array([0,np.cos(0.98 * t)])
        r = lambda t: np.array([np.cos(0.5 * t), np.sin(0.5 * t)])
        rT = np.array([0.1, 0])

        clqt = clqt_np.CLQT(F, L, X, U, XT, c, H, r, HT, rT, T)
        t = 1.0
        return clqt

    def test_backwardPass_1(self):
        clqt, _ = self.getSimpleCLQT(10.0)

        steps = 10
        blocks = 10
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(blocks * steps)

        para = para_np.Parareal_CLQT_np(clqt)

        for i in range(11):
            if i == 0:
                S_curr_list, v_curr_list = para.initBackwardPass(blocks)
                S_G_list = S_curr_list
                v_G_list = v_curr_list
            else:
                S_F_list, v_F_list = para.denseBackwardPass(steps, S_curr_list, v_curr_list)
                S_curr_list, v_curr_list, S_G_list, v_G_list = \
                    para.coarseBackwardPass(S_F_list, v_F_list, S_G_list, v_G_list)

            Kx_list2, d_list2, S_list2, v_list2 = para.finalBackwardPass(steps, S_curr_list, v_curr_list)

            err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
            print(f"Iteration {i} : {err}")

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)

    def test_backwardPass_2(self):
        clqt, _ = self.getSimpleCLQT(10.0)

        steps = 20
        blocks = 10
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(blocks * steps)
        para = para_np.Parareal_CLQT_np(clqt)

        Kx_list2, d_list2, S_list2, v_list2 = para.backwardPass(blocks, steps, steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

    def test_backwardPass_3(self):
        clqt = self.getRandomCLQT(10.0)

        steps = 20
        blocks = 15
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(blocks * steps)
        para = para_np.Parareal_CLQT_np(clqt)

        Kx_list2, d_list2, S_list2, v_list2 = para.backwardPass(blocks, steps, steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        print(err)
        self.assertTrue(err < 1e-5)

    def test_forwardPass_1(self):
        clqt, _ = self.getSimpleCLQT(5.0)
        x0 = np.array([1,2,3])

        steps = 10
        blocks = 10
        Kx_list, d_list, S_list, v_list = clqt.seqBackwardPass(blocks * steps)

        u_zoh = False
#        u_zoh = True
        u_list1, x_list1 = clqt.seqForwardPass(x0, Kx_list, d_list, u_zoh=u_zoh)

        para = para_np.Parareal_CLQT_np(clqt)

        for i in range(11):
            if i == 0:
                x_curr_list = para.initForwardPass(blocks, x0, Kx_list, d_list, u_zoh=u_zoh)
                x_G_list = x_curr_list
            else:
                x_F_list = para.denseForwardPass(steps, x_curr_list, Kx_list, d_list, u_zoh=u_zoh)
                x_curr_list, x_G_list = \
                    para.coarseForwardPass(x_F_list, x_G_list, Kx_list, d_list, u_zoh=u_zoh)

            u_list2, x_list2 = para.finalForwardPass(steps, x_curr_list, Kx_list, d_list, u_zoh=u_zoh)

            err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
            print(f"Iteration {i} : {err}")

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-5)

    def test_forwardPass_2(self):
        clqt, _ = self.getSimpleCLQT(5.0)
        x0 = np.array([1, 2, 3])

        steps = 10
        blocks = 10
        Kx_list, d_list, S_list, v_list = clqt.seqBackwardPass(blocks * steps)

        u_zoh = True
        u_list1, x_list1 = clqt.seqForwardPass(x0, Kx_list, d_list, u_zoh=u_zoh)

        para = para_np.Parareal_CLQT_np(clqt)
        u_list2, x_list2 = para.forwardPass(blocks, steps, x0, Kx_list, d_list, u_zoh=u_zoh)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-5)

    def test_forwardPass_3(self):
        clqt = self.getRandomCLQT(10.0)

        x0 = np.array([2,3])

        steps = 20
        blocks = 15
        Kx_list, d_list, S_list, v_list = clqt.seqBackwardPass(blocks * steps)

        u_zoh = True
        u_list1, x_list1 = clqt.seqForwardPass(x0, Kx_list, d_list, u_zoh=u_zoh)

        para = para_np.Parareal_CLQT_np(clqt)
        u_list2, x_list2 = para.forwardPass(blocks, steps, x0, Kx_list, d_list, u_zoh=u_zoh)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-5)

    def test_fwdbwdPass_1(self):
        clqt, _ = self.getSimpleCLQT(5.0)
        x0 = np.array([1,2,3])

        steps = 10
        blocks = 10
        A_list1, b_list1, C_list1 = clqt.seqFwdBwdPass(x0, blocks * steps)

        para = para_np.Parareal_CLQT_np(clqt)

        for i in range(11):
            if i == 0:
                A_curr_list, b_curr_list, C_curr_list = para.initFwdBwdPass(blocks, x0)
                A_G_list = A_curr_list
                b_G_list = b_curr_list
                C_G_list = C_curr_list
            else:
                A_F_list, b_F_list, C_F_list = para.denseFwdBwdPass(steps, A_curr_list, b_curr_list, C_curr_list)
                A_curr_list, b_curr_list, C_curr_list, A_G_list, b_G_list, C_G_list = \
                    para.coarseFwdBwdPass(A_F_list, b_F_list, C_F_list, A_G_list, b_G_list, C_G_list)

            A_list2, b_list2, C_list2 = para.finalFwdBwdPass(steps, A_curr_list, b_curr_list, C_curr_list)

            err = max([linalg.norm(e1 - e2) for e1, e2 in zip(b_list1, b_list2)])
            print(f"Iteration {i} : {err}")

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(A_list1, A_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(b_list1, b_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(C_list1, C_list2)])
        self.assertTrue(err < 1e-5)

    def test_fwdBwdPass_2(self):
        clqt, _ = self.getSimpleCLQT(5.0)
        x0 = np.array([1, 2, 3])

        steps = 10
        blocks = 10
        A_list1, b_list1, C_list1 = clqt.seqFwdBwdPass(x0, blocks * steps)

        para = para_np.Parareal_CLQT_np(clqt)
        A_list2, b_list2, C_list2 = para.fwdBwdPass(blocks, steps, x0)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(A_list1, A_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(b_list1, b_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(C_list1, C_list2)])
        self.assertTrue(err < 1e-5)

    def test_fwdBwdPass_3(self):
        clqt = self.getRandomCLQT(10.0)

        x0 = np.array([2,3])

        steps = 20
        blocks = 15

        A_list1, b_list1, C_list1 = clqt.seqFwdBwdPass(x0, blocks * steps)

        para = para_np.Parareal_CLQT_np(clqt)
        A_list2, b_list2, C_list2 = para.fwdBwdPass(blocks, steps, x0)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(A_list1, A_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(b_list1, b_list2)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(C_list1, C_list2)])
        self.assertTrue(err < 1e-5)
