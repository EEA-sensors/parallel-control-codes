#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for TF version of parareal CLQT.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import tensorflow as tf
import parallel_control.clqt_np as clqt_np
import parallel_control.clqt_tf as clqt_tf
import parallel_control.parareal_clqt_np as para_np
import parallel_control.parareal_clqt_tf as para_tf
import math
import unittest

import matplotlib.pyplot as plt

##############################################################################
# Unit tests for TF Parareal CLQR
##############################################################################

class Parareal_CLQR_tf_UnitTest(unittest.TestCase):
    """Unit tests for TF Parareal CLQR"""

    def getCLQT(self,T=10.0):
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

        dp = lambda v: tf.constant(v, dtype=tf.float64)

        X_f = lambda t: (dp(0.02) + dp(0.01) * tf.sin(dp(1.0) * t)) * tf.eye(2, dtype=tf.float64)
        U_f = lambda t: (dp(0.04) + dp(0.01) * tf.cos(dp(0.3) * t)) * tf.eye(1, dtype=tf.float64)
        XT = tf.constant(XT, dtype=tf.float64)
        F_f = lambda t: tf.stack([tf.stack([dp(0.0), dp(1.0)]), tf.stack([-(dp(2.0) * tf.sin(dp(3.1) * t)) ** 2, dp(0.0)])])
#        L_f = lambda t: tf.stack([tf.stack([dp(0.0)]), tf.stack([dp(1.0) + dp(0.1) * tf.sin(dp(2.1) * t)])])
        def L_f(t):
            ta = tf.TensorArray(tf.float64, size=2, dynamic_size=False, infer_shape=True)
            ta = ta.write(0, tf.constant([0.0], dtype=tf.float64))
            ta = ta.write(1, tf.expand_dims(dp(1.0) + 0.1 * tf.sin(2.1 * t), -1))
            return ta.stack()

        HT = tf.constant(HT, dtype=tf.float64)
        H_f = lambda t: tf.stack([tf.stack([dp(1.0),dp(0.0)]), tf.stack([dp(0.1),dp(1.0)])])
        c_f = lambda t: tf.stack([dp(0.0), tf.cos(dp(0.98) * t)])
        r_f = lambda t: tf.stack([tf.cos(dp(0.5) * t), tf.sin(dp(0.5) * t)])
        rT = tf.constant(rT, dtype=tf.float64)
        T  = tf.constant(clqt.T, dtype=tf.float64)

        x0 = np.array([1,2])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        return clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f


    def test_init_bwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10
        S_curr_list1, v_curr_list1 = para.initBackwardPass(blocks)
        Ss1 = tf.convert_to_tensor(S_curr_list1, dtype=tf.float64)
        vs1 = tf.convert_to_tensor(v_curr_list1, dtype=tf.float64)

        Ss2, vs2 = para_tf.pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_dense_bwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10
        S_curr_list1, v_curr_list1 = para.initBackwardPass(blocks)
        S_F_list1, v_F_list1 = para.denseBackwardPass(steps, S_curr_list1, v_curr_list1)
        Ss_F1 = tf.convert_to_tensor(S_F_list1, dtype=tf.float64)
        vs_F1 = tf.convert_to_tensor(v_F_list1, dtype=tf.float64)

        Ss2, vs2 = para_tf.pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss_F2, vs_F2 = para_tf.pclqt_dense_backwardpass(steps, T, Ss2, vs2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(Ss_F1 - Ss_F2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs_F1 - vs_F2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_coarse_bwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10
        S_curr_list1, v_curr_list1 = para.initBackwardPass(blocks)
        S_G_list1 = S_curr_list1
        v_G_list1 = v_curr_list1
        S_F_list1, v_F_list1 = para.denseBackwardPass(steps, S_curr_list1, v_curr_list1)
        S_curr_list1, v_curr_list1, S_G_list1, v_G_list1 = \
            para.coarseBackwardPass(S_F_list1, v_F_list1, S_G_list1, v_G_list1)

        Ss1 = tf.convert_to_tensor(S_curr_list1, dtype=tf.float64)
        vs1 = tf.convert_to_tensor(v_curr_list1, dtype=tf.float64)

        Ss2, vs2 = para_tf.pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss_G2 = Ss2
        vs_G2 = vs2
        Ss_F2, vs_F2 = para_tf.pclqt_dense_backwardpass(steps, T, Ss2, vs2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss2, vv2, Ss_G2, vs_G2 = para_tf.pclqt_coarse_backwardpass(T, Ss_F2, vs_F2, Ss_G2, vs_G2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_final_bwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10
        S_curr_list1, v_curr_list1 = para.initBackwardPass(blocks)
        S_G_list1 = S_curr_list1
        v_G_list1 = v_curr_list1
        S_F_list1, v_F_list1 = para.denseBackwardPass(steps, S_curr_list1, v_curr_list1)
        S_curr_list1, v_curr_list1, S_G_list1, v_G_list1 = \
            para.coarseBackwardPass(S_F_list1, v_F_list1, S_G_list1, v_G_list1)
        Kx_list1, d_list1, S_list1, v_list1 = para.finalBackwardPass(steps, S_curr_list1, v_curr_list1)

        Ss1 = tf.convert_to_tensor(S_list1, dtype=tf.float64)
        vs1 = tf.convert_to_tensor(v_list1, dtype=tf.float64)
        Kxs1 = tf.convert_to_tensor(Kx_list1, dtype=tf.float64)
        ds1 = tf.convert_to_tensor(d_list1, dtype=tf.float64)

        Ss2, vs2 = para_tf.pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss_G2 = Ss2
        vs_G2 = vs2
        Ss_F2, vs_F2 = para_tf.pclqt_dense_backwardpass(steps, T, Ss2, vs2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss2, vs2, Ss_G2, vs_G2 = para_tf.pclqt_coarse_backwardpass(T, Ss_F2, vs_F2, Ss_G2, vs_G2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_final_backwardpass(steps, T, Ss2, vs2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_bwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss1 = tf.convert_to_tensor(S_list1, dtype=tf.float64)
        vs1 = tf.convert_to_tensor(v_list1, dtype=tf.float64)
        Kxs1 = tf.convert_to_tensor(Kx_list1, dtype=tf.float64)
        ds1 = tf.convert_to_tensor(d_list1, dtype=tf.float64)

        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)


    def test_bwpass_2(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)
        blocks = 5000 # This appears to fail
        steps = 10

        niter = steps

        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    def test_init_fwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        u_zoh = False
#        u_zoh = True
        x_curr_list1 = para.initForwardPass(blocks, x0, Kx_list1, d_list1, u_zoh=u_zoh)
        xs_curr1 = tf.convert_to_tensor(x_curr_list1, dtype=tf.float64)

        xs_curr2 = para_tf.pclqt_init_forwardpass(blocks, steps, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)

        err = tf.reduce_max(tf.math.abs(xs_curr1 - xs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_dense_fwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        u_zoh = False
#        u_zoh = True
        x_curr_list1 = para.initForwardPass(blocks, x0, Kx_list1, d_list1, u_zoh=u_zoh)
        x_F_list1 = para.denseForwardPass(steps, x_curr_list1, Kx_list1, d_list1, u_zoh=u_zoh)
        xs_F1 = tf.convert_to_tensor(x_F_list1, dtype=tf.float64)

        xs_curr2 = para_tf.pclqt_init_forwardpass(blocks, steps, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_F2 = para_tf.pclqt_dense_forwardpass(steps, T, xs_curr2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)

        err = tf.reduce_max(tf.math.abs(xs_F1 - xs_F2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_coarse_fwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        u_zoh = False
#        u_zoh = True
        x_curr_list1 = para.initForwardPass(blocks, x0, Kx_list1, d_list1, u_zoh=u_zoh)
        x_G_list1 = x_curr_list1
        x_F_list1 = para.denseForwardPass(steps, x_curr_list1, Kx_list1, d_list1, u_zoh=u_zoh)
        x_curr_list1, x_G_list1 = \
            para.coarseForwardPass(x_F_list1, x_G_list1, Kx_list1, d_list1, u_zoh=u_zoh)
        xs_curr1 = tf.convert_to_tensor(x_curr_list1, dtype=tf.float64)
        xs_G1 = tf.convert_to_tensor(x_G_list1, dtype=tf.float64)

        xs_curr2 = para_tf.pclqt_init_forwardpass(blocks, steps, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_G2 = xs_curr2
        xs_F2 = para_tf.pclqt_dense_forwardpass(steps, T, xs_curr2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_curr2, xs_G2 = para_tf.pclqt_coarse_forwardpass(steps, T, xs_F2, xs_G2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)

        err = tf.reduce_max(tf.math.abs(xs_curr1 - xs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs_G1 - xs_G2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_final_fwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        u_zoh = False
#        u_zoh = True
        x_curr_list1 = para.initForwardPass(blocks, x0, Kx_list1, d_list1, u_zoh=u_zoh)
        x_G_list1 = x_curr_list1
        x_F_list1 = para.denseForwardPass(steps, x_curr_list1, Kx_list1, d_list1, u_zoh=u_zoh)
        x_curr_list1, x_G_list1 = \
            para.coarseForwardPass(x_F_list1, x_G_list1, Kx_list1, d_list1, u_zoh=u_zoh)
        u_list1, x_list1 = para.finalForwardPass(steps, x_curr_list1, Kx_list1, d_list1, u_zoh=u_zoh)

        xs1 = tf.convert_to_tensor(x_list1, dtype=tf.float64)
        us1 = tf.convert_to_tensor(u_list1, dtype=tf.float64)

        xs_curr2 = para_tf.pclqt_init_forwardpass(blocks, steps, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_G2 = xs_curr2
        xs_F2 = para_tf.pclqt_dense_forwardpass(steps, T, xs_curr2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_curr2, xs_G2 = para_tf.pclqt_coarse_forwardpass(steps, T, xs_F2, xs_G2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)
        xs2, us2 = para_tf.pclqt_final_forwardpass(steps, T, xs_curr2, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_fwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        u_zoh = False
#        u_zoh = True

        niter = 2
        u_list1, x_list1 = para.forwardPass(blocks, steps, x0, Kx_list1, d_list1, u_zoh=u_zoh, niter=niter)
        xs1 = tf.convert_to_tensor(x_list1, dtype=tf.float64)
        us1 = tf.convert_to_tensor(u_list1, dtype=tf.float64)

        xs2, us2 = para_tf.pclqt_forwardpass(blocks, steps, niter, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f, u_zoh=u_zoh)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_bwfwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)

        u_zoh = False
#        u_zoh = True

        niter = 2
        u_list1, x_list1 = para.forwardPass(blocks, steps, x0, Kx_list1, d_list1, u_zoh=u_zoh, niter=niter)
        xs1 = tf.convert_to_tensor(x_list1, dtype=tf.float64)
        us1 = tf.convert_to_tensor(u_list1, dtype=tf.float64)

        xs2, us2 = para_tf.pclqt_backwardforwardpass(blocks, steps, niter, x0_tf, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        xs2, us2 = para_tf.pclqt_backwardforwardpass(blocks, steps, niter, x0_tf, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_init_fwdbwdpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        A_curr_list1, b_curr_list1, C_curr_list1 = para.initFwdBwdPass(blocks, x0)
        As_curr1 = tf.convert_to_tensor(A_curr_list1, dtype=tf.float64)
        bs_curr1 = tf.convert_to_tensor(b_curr_list1, dtype=tf.float64)
        Cs_curr1 = tf.convert_to_tensor(C_curr_list1, dtype=tf.float64)

        As_curr2, bs_curr2, Cs_curr2 = para_tf.pclqt_init_fwdbwdpass(blocks, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(As_curr1 - As_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs_curr1 - bs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs_curr1 - Cs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_dense_fwdbwdpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        A_curr_list1, b_curr_list1, C_curr_list1 = para.initFwdBwdPass(blocks, x0)
        A_F_list1, b_F_list1, C_F_list1 = para.denseFwdBwdPass(steps, A_curr_list1, b_curr_list1, C_curr_list1)

        As_curr2, bs_curr2, Cs_curr2 = para_tf.pclqt_init_fwdbwdpass(blocks, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_F2, bs_F2, Cs_F2 = para_tf.pclqt_dense_fwdbwdpass(steps, T, As_curr2, bs_curr2, Cs_curr2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        As_F1 = tf.convert_to_tensor(A_F_list1, dtype=tf.float64)
        bs_F1 = tf.convert_to_tensor(b_F_list1, dtype=tf.float64)
        Cs_F1 = tf.convert_to_tensor(C_F_list1, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As_F1 - As_F2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs_F1 - bs_F2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs_F1 - Cs_F2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_coarse_fwdbwdpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        A_curr_list1, b_curr_list1, C_curr_list1 = para.initFwdBwdPass(blocks, x0)
        A_G_list1 = A_curr_list1
        b_G_list1 = b_curr_list1
        C_G_list1 = C_curr_list1
        A_F_list1, b_F_list1, C_F_list1 = para.denseFwdBwdPass(steps, A_curr_list1, b_curr_list1, C_curr_list1)
        A_curr_list1, b_curr_list1, C_curr_list1, A_G_list1, b_G_list1, C_G_list1 = \
            para.coarseFwdBwdPass(A_F_list1, b_F_list1, C_F_list1, A_G_list1, b_G_list1, C_G_list1)
        As_curr1 = tf.convert_to_tensor(A_curr_list1, dtype=tf.float64)
        bs_curr1 = tf.convert_to_tensor(b_curr_list1, dtype=tf.float64)
        Cs_curr1 = tf.convert_to_tensor(C_curr_list1, dtype=tf.float64)
        As_G1 = tf.convert_to_tensor(A_G_list1, dtype=tf.float64)
        bs_G1 = tf.convert_to_tensor(b_G_list1, dtype=tf.float64)
        Cs_G1 = tf.convert_to_tensor(C_G_list1, dtype=tf.float64)

        As_curr2, bs_curr2, Cs_curr2 = para_tf.pclqt_init_fwdbwdpass(blocks, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_G2 = As_curr2
        bs_G2 = bs_curr2
        Cs_G2 = Cs_curr2
        As_F2, bs_F2, Cs_F2 = para_tf.pclqt_dense_fwdbwdpass(steps, T, As_curr2, bs_curr2, Cs_curr2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_curr2, bs_curr2, Cs_curr2, As_G2, bs_G2, Cs_G2 = \
            para_tf.pclqt_coarse_fwdbwdpass(T, As_F2, bs_F2, Cs_F2, As_G2, bs_G2, Cs_G2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(As_curr1 - As_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs_curr1 - bs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs_curr1 - Cs_curr2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(As_G1 - As_G2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs_G1 - bs_G2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs_G1 - Cs_G2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_final_fwdbwdpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        A_curr_list1, b_curr_list1, C_curr_list1 = para.initFwdBwdPass(blocks, x0)
        A_G_list1 = A_curr_list1
        b_G_list1 = b_curr_list1
        C_G_list1 = C_curr_list1
        A_F_list1, b_F_list1, C_F_list1 = para.denseFwdBwdPass(steps, A_curr_list1, b_curr_list1, C_curr_list1)
        A_curr_list1, b_curr_list1, C_curr_list1, A_G_list1, b_G_list1, C_G_list1 = \
            para.coarseFwdBwdPass(A_F_list1, b_F_list1, C_F_list1, A_G_list1, b_G_list1, C_G_list1)
        A_list1, b_list1, C_list1 = para.finalFwdBwdPass(steps, A_curr_list1, b_curr_list1, C_curr_list1)

        As1 = tf.convert_to_tensor(A_list1, dtype=tf.float64)
        bs1 = tf.convert_to_tensor(b_list1, dtype=tf.float64)
        Cs1 = tf.convert_to_tensor(C_list1, dtype=tf.float64)

        As_curr2, bs_curr2, Cs_curr2 = para_tf.pclqt_init_fwdbwdpass(blocks, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_G2 = As_curr2
        bs_G2 = bs_curr2
        Cs_G2 = Cs_curr2
        As_F2, bs_F2, Cs_F2 = para_tf.pclqt_dense_fwdbwdpass(steps, T, As_curr2, bs_curr2, Cs_curr2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_curr2, bs_curr2, Cs_curr2, As_G2, bs_G2, Cs_G2 = \
            para_tf.pclqt_coarse_fwdbwdpass(T, As_F2, bs_F2, Cs_F2, As_G2, bs_G2, Cs_G2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        As2, bs2, Cs2 = para_tf.pclqt_final_fwdbwdpass(steps, T, As_curr2, bs_curr2, Cs_curr2, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_fwdbwdpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
        A_list1, b_list1, C_list1 = para.fwdBwdPass(blocks, steps, x0, niter=niter)
        As1 = tf.convert_to_tensor(A_list1, dtype=tf.float64)
        bs1 = tf.convert_to_tensor(b_list1, dtype=tf.float64)
        Cs1 = tf.convert_to_tensor(C_list1, dtype=tf.float64)

        As2, bs2, Cs2 = para_tf.pclqt_fwdbwdpass(blocks, steps, niter, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_bwbwfwpass(self):
        clqt, x0, x0_tf, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getCLQT(T=1.0)

        para = para_np.Parareal_CLQT_np(clqt)

        blocks = 12
        steps = 10

        niter = steps
#        niter = 2

        Kx_list1, d_list1, S_list1, v_list1 = para.backwardPass(blocks, steps, niter=niter)
        A_list1, b_list1, C_list1 = para.fwdBwdPass(blocks, steps, x0)
        u_list1, x_list1 = clqt.combineForwardBackward(Kx_list1, d_list1, S_list1, v_list1, A_list1, b_list1, C_list1)

        xs1 = tf.convert_to_tensor(x_list1, dtype=tf.float64)
        us1 = tf.convert_to_tensor(u_list1, dtype=tf.float64)

        xs2, us2 = para_tf.pclqt_backwardbwdfwdpass(blocks, steps, niter, x0_tf, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

