#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for linear speedtests.

@author: Simo Särkkä
"""

import unittest
import tensorflow as tf
import parallel_control.lqt_tf as lqt_tf
import parallel_control.linear_model_np as linear_model_np
import parallel_control.mass_model_np as mass_model_np
import parallel_control.linear_speedtest as linspeed
import math

class LinearSpeedtest_UnitTest(unittest.TestCase):
    def getTrackerLQT(self):
        T = 100
        dtype = tf.float64
        model = linear_model_np.LinearModel()
        xy = model.genData(T // 10)
        lqt, x0 = model.getLQT(xy)
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt, dtype=dtype)
        x0_tf  = tf.convert_to_tensor(x0, dtype=dtype)
        return (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

    def test_tracking_sequential_bw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        Ss, vs, Kxs, ds = linspeed.lqt_sequential_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_sequential_bw_speedtest(lqt_gen)

    def test_tracking_parallel_bw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        Ss, vs, Kxs, ds = linspeed.lqt_parallel_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_parallel_bw_speedtest(lqt_gen)

    def test_tracking_sequential_bwfw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_sequential_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_sequential_bwfw_speedtest(lqt_gen)

    def test_tracking_parallel_bwfw_1(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_parallel_bwfw_1_speedtest(lqt_gen)

    def test_tracking_parallel_bwfw_2(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_parallel_bwfw_2_speedtest(lqt_gen)

    def test_tracking_sequential_cond_bwfw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_sequential_cond_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_sequential_cond_bwfw_speedtest(lqt_gen)

    def test_tracking_parallel_cond_bwfw_1(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_cond_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_parallel_cond_bwfw_1_speedtest(lqt_gen)

    def test_tracking_parallel_cond_bwfw_2(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getTrackerLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_cond_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.tracking_generator(2, 3, 3)
        _ = linspeed.lqt_parallel_cond_bwfw_2_speedtest(lqt_gen)

    def test_tracking_sequential_cond_bwfw_single_block(self):
        lqt_gen = linspeed.tracking_generator(2, 2, 1)
        _ = linspeed.lqt_sequential_cond_bwfw_speedtest(lqt_gen, Nc=256)

        lqt_gen = linspeed.tracking_generator(2, 2, 1)
        _ = linspeed.lqt_parallel_cond_bwfw_1_speedtest(lqt_gen, Nc=256)

        lqt_gen = linspeed.tracking_generator(2, 2, 1)
        _ = linspeed.lqt_parallel_cond_bwfw_2_speedtest(lqt_gen, Nc=256)

    def getMassLQT(self):
        T = 100
        dtype = tf.float64
        model = mass_model_np.MassModel(7)
        Tf = 10.0
        dt = Tf / T
        lqt, x0 = model.getLQT(dt=dt, Tf=Tf)
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt, dtype=dtype)
        x0_tf  = tf.convert_to_tensor(x0, dtype=dtype)
        return (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

    def test_mass_sequential_bw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        Ss, vs, Kxs, ds = linspeed.lqt_sequential_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_sequential_bw_speedtest(lqt_gen)

    def test_mass_parallel_bw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        Ss, vs, Kxs, ds = linspeed.lqt_parallel_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(S_list, dtype=Ss.dtype) - Ss))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_list, dtype=vs.dtype) - vs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(Kx_list, dtype=Kxs.dtype) - Kxs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(d_list, dtype=ds.dtype) - ds))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_parallel_bw_speedtest(lqt_gen)

    def test_mass_sequential_bwfw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_sequential_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_sequential_bwfw_speedtest(lqt_gen)

    def test_mass_parallel_bwfw_1(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_parallel_bwfw_1_speedtest(lqt_gen)

    def test_mass_parallel_bwfw_2(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_parallel_bwfw_2_speedtest(lqt_gen)

    def test_mass_sequential_cond_bwfw(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_sequential_cond_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_sequential_cond_bwfw_speedtest(lqt_gen)

    def test_mass_parallel_cond_bwfw_1(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_cond_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_parallel_cond_bwfw_1_speedtest(lqt_gen)

    def test_mass_parallel_cond_bwfw_2(self):
        (x0, lqt, x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) = self.getMassLQT()
        Kx_list, d_list, S_list, v_list = lqt.seqBackwardPass()
        u_list, x_list = lqt.seqForwardPass(x0, Kx_list, d_list)
        xs, us = linspeed.lqt_parallel_cond_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, 6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list, dtype=us.dtype) - us))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list, dtype=xs.dtype) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

        lqt_gen = linspeed.mass_generator(6, 2, 3, 3)
        _ = linspeed.lqt_parallel_cond_bwfw_2_speedtest(lqt_gen)

    def test_mass_gen_2(self):
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_sequential_bw_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_parallel_bw_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_sequential_bwfw_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_parallel_bwfw_1_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_parallel_bwfw_2_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_sequential_cond_bwfw_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_parallel_cond_bwfw_1_speedtest(lqt_gen)
        lqt_gen = linspeed.mass_generator_2(100, nstart=2, nend=4, nstep=1)
        _ = linspeed.lqt_parallel_cond_bwfw_2_speedtest(lqt_gen)
