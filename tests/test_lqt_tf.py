"""
Unit tests for TensorFlow-based Linear Quadratic Tracker (LQT) routines.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf

import unittest
import parallel_control.lqt_np as lqt_np
import parallel_control.lqt_tf as lqt_tf

class LQT_tf_UnitTest(unittest.TestCase):

    def setupRndLQT(self, T=5):
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
            #s[i] = rng.standard_normal(nu)
            r[i] = rng.standard_normal(nr)
            #M[i] = rng.standard_normal((nr, nu))
            #Z[i] = rng.standard_normal((nu, nu))
            H[i] = rng.standard_normal((nr, nx))
            L[i] = rng.standard_normal((nx, nu))
            X[i] = rng.standard_normal((nr, 2 * nr))
            X[i] = X[i] @ X[i].T
            U[i] = rng.standard_normal((nu, 2 * nu))
            U[i] = U[i] @ U[i].T + np.eye(nu)

        lqt = lqt_np.LQT.checkAndExpand(F,L,X,U,XT,c,H,r,HT,rT,None,None,None,T)

        return lqt, x0

    def setupLQT_tf_inputs(self, lqt):
        return lqt_tf.lqt_np_to_tf(lqt)

    def test_sequential(self):
        lqt, x0 = self.setupRndLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = self.setupLQT_tf_inputs(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()

        Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        print(Kx_list1)
        print(Kxs)


        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list1, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list1, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list1, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list1, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)

        u_list1, x_list1 = lqt.seqForwardPass(x0,Kx_list1,d_list1)

        xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list1, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list1, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_parallel_1(self):
        lqt, x0 = self.setupRndLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = self.setupLQT_tf_inputs(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()

        Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list1, dtype=Ss.dtype) - Ss))
        self.assertTrue(err < 1e-10)
        print(err)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list1, dtype=vs.dtype) - vs))
        self.assertTrue(err < 1e-10)
        print(err)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list1, dtype=Kxs.dtype) - Kxs))
        self.assertTrue(err < 1e-10)
        print(err)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list1, dtype=ds.dtype) - ds))
        self.assertTrue(err < 1e-10)
        print(err)

        u_list1, x_list1 = lqt.seqForwardPass(x0,Kx_list1,d_list1)

        xs, us = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list1, dtype=xs.dtype) - xs))
        self.assertTrue(err < 1e-10)
        print(err)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list1, dtype=us.dtype) - us))
        self.assertTrue(err < 1e-10)
        print(err)

    def test_parallel_2(self):
        lqt, x0 = self.setupRndLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = self.setupLQT_tf_inputs(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()

        Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        u_list1, x_list1 = lqt.seqForwardPass(x0,Kx_list1,d_list1)

        xs, us = lqt_tf.lqt_par_fwdbwdpass(x0_tf, Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list1, dtype=xs.dtype) - xs))
        self.assertTrue(err < 1e-10)
        print(err)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list1, dtype=us.dtype) - us))
        self.assertTrue(err < 1e-10)
        print(err)

    def test_single_step(self):
        lqt, x0 = self.setupRndLQT(1)
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = self.setupLQT_tf_inputs(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)

        Ss0, vs0, Kxs0, ds0 = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs0, us0 = lqt_tf.lqt_seq_forwardpass(x0_tf, Fs, cs, Ls, Kxs0, ds0)

        Ss1, vs1, Kxs1, ds1 = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs1, us1 = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs1, ds1)
        xs2, us2 = lqt_tf.lqt_par_fwdbwdpass(x0_tf, Fs, cs, Ls, Hs, rs, Xs, Us, Ss1, vs1, Kxs1, ds1)

        print("From single step:")
        print(us1)

        err = tf.math.reduce_max(tf.math.abs(Ss1 - Ss0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(vs1 - vs0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(Kxs1 - Kxs0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(ds1 - ds0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(xs1 - xs0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(xs2 - xs0))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us2 - us0))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_cost(self):
        lqt, x0 = self.setupRndLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = self.setupLQT_tf_inputs(lqt)

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)

        us = tf.convert_to_tensor(u_list1)
        xs = tf.convert_to_tensor(x_list1)

        cost1 = lqt.cost(u_list1, x_list1)
        cost2 = lqt_tf.lqt_cost(xs, us, Hs, HT, rs, rT, Xs, XT, Us)

        self.assertTrue(tf.abs(cost2 - cost1) < 1e-10)

        print(cost1)
        print(cost2)

    def setupGenRndLQT(self, T=5):
        #
        # Test by initializing with random matrices
        # while keeping the sizes the default
        #
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

        lqt = lqt_np.LQT.checkAndExpand(F,L,X,U,XT,c,H,r,HT,rT,None,s,M,T)

        return lqt, x0

    def test_transform(self):
        lqt, x0 = self.setupGenRndLQT()
        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Ms, ss = lqt_tf.lqt_np_to_tf_gen(lqt)

        Fs_trans, cs_trans, Xs_trans = lqt_tf.lqt_gen_to_canon(Fs, cs, Ls, Hs, rs, Xs, Us, Ms, ss)
        Ss, vs, Kxs_trans, ds_trans = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Ls, Hs, HT, rs, rT, Xs_trans, XT, Us)
        Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hs, rs, Us, Ms, ss)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list1, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list1, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list1, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list1, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)
