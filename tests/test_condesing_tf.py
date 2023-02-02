#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for TensorFlow version of partial condensing for LQT.

@author: Simo Särkkä
"""

import unittest
from scipy import linalg
import numpy as np
import pprint
import parallel_control.lqt_np as lqt_np
import parallel_control.lqt_tf as lqt_tf
import parallel_control.condensing_np as condensing_np
import parallel_control.condensing_tf as condensing_tf

import tensorflow as tf
import math

mm = tf.linalg.matmul
mv = tf.linalg.matvec


class LQTCondenser_tf_UnitTest(unittest.TestCase):
    def setupLQR(self,T):
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

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.F, dtype=Fstar.dtype) - Fstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.c, dtype=cstar.dtype) - cstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.L, dtype=Lstar.dtype) - Lstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.H, dtype=Hstar.dtype) - Hstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.r, dtype=rstar.dtype) - rstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.X, dtype=Xstar.dtype) - Xstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.U, dtype=Ustar.dtype) - Ustar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.M, dtype=Mstar.dtype) - Mstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.s, dtype=sstar.dtype) - sstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.Lambda_list, dtype=Lambda.dtype) - Lambda))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.cbar_list, dtype=cbar.dtype) - cbar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.Lbar_list, dtype=Lbar.dtype) - Lbar))
        self.assertTrue(err < 1e-10)

    def setupRndLQT(self,T=11):
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

    def test_condensing_2(self):
        lqt, x0 = self.setupRndLQT()

        cond = condensing_np.LQTCondenser()
        Nc = 3
        clqt = cond.condense(lqt, Nc)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.F, dtype=Fstar.dtype) - Fstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.c, dtype=cstar.dtype) - cstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.L, dtype=Lstar.dtype) - Lstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.H, dtype=Hstar.dtype) - Hstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.r, dtype=rstar.dtype) - rstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.X, dtype=Xstar.dtype) - Xstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.U, dtype=Ustar.dtype) - Ustar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.M, dtype=Mstar.dtype) - Mstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(clqt.s, dtype=sstar.dtype) - sstar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.Lambda_list, dtype=Lambda.dtype) - Lambda))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.cbar_list, dtype=cbar.dtype) - cbar))
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(cond.Lbar_list, dtype=Lbar.dtype) - Lbar))
        self.assertTrue(err < 1e-10)

    def test_condensing_3(self):
        lqt, x0 = self.setupRndLQT()

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)
        Nc = 3

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        Fs_trans, cs_trans, Xs_trans = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)
        Ss, vs, Kxs_trans, ds_trans = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar)
        Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

        xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds)

        us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

        Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs2, us2 = lqt_tf.lqt_seq_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_condensing_4(self):
        lqt, x0 = self.setupRndLQT()

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)
        Nc = 3

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        Fs_trans, cs_trans, Xs_trans = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)
        Ss, vs, Kxs_trans, ds_trans = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar)
        Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

        xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds)

        us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

        Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs2, us2 = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_condensing_single_block_1(self):
        lqt, x0 = self.setupRndLQT(3)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)
        Nc = 3

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        Fs_trans, cs_trans, Xs_trans = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)
        Ss, vs, Kxs_trans, ds_trans = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar)
        Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

        xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds)

        us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

        Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs2, us2 = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_condensing_single_block_2(self):
        lqt, x0 = self.setupRndLQT(3)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt)
        x0_tf = tf.convert_to_tensor(x0, dtype=Fs.dtype)
        Nc = 5

        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

        Fs_trans, cs_trans, Xs_trans = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)
        Ss, vs, Kxs_trans, ds_trans = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar)
        Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

        xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds)

        us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

        Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
        xs2, us2 = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)

        err = tf.math.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-10)
