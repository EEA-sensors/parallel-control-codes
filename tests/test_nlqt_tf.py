#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Tensorflow-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf

import parallel_control.lqt_np as lqt_np
import parallel_control.lqt_tf as lqt_tf
import parallel_control.nlqt_np as nlqt_np
import parallel_control.nlqt_tf as nlqt_tf
import test_nlqt_np
import unittest


def nlqt_tf_test_f(xs, us):
    return -tf.sin(xs) + us

def nlqt_tf_test_Fx(xs, us):
    return tf.expand_dims(-tf.cos(xs),-1)

def nlqt_tf_test_Fu(xs, us):
    return tf.expand_dims(tf.ones_like(xs),-1)


class NLQT_tf_UnitTest(unittest.TestCase):
    def setupNLQT(self):
        model = test_nlqt_np.NLQT_np_testmodel()

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
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(nlqt.lqt)

        xs = tf.convert_to_tensor(x)
        us = tf.convert_to_tensor(u)

        Fs2, cs2, Ls2 = nlqt_tf.nlqt_linearize(us, xs[:-1, ...], nlqt_tf_test_f, nlqt_tf_test_Fx, nlqt_tf_test_Fu)

#        print(Fs2)
#        print(cs2)
#        print(Ls2)

        err = tf.math.reduce_max(tf.math.abs(Fs2 - Fs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(cs2 - cs))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(Ls2 - Ls))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_simulate(self):
        nlqt, x, u, r = self.setupNLQT()
        x_list = nlqt.simulate(x[0], u)
        us = tf.convert_to_tensor(u)
        xs = nlqt_tf.nlqt_simulate(x[0], us, nlqt_tf_test_f)

        print(x_list)
        print(xs)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(x_list) - xs))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_cost(self):
        nlqt, x, u, r = self.setupNLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(nlqt.lqt)

        cost1 = nlqt.cost(x[0], u)
        print("cost1 = %f" % cost1)
        us = tf.convert_to_tensor(u)
        cost2 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
        print("cost2 = %f" % cost2)
        self.assertTrue(tf.math.abs(cost2 - cost1) < 1e-10)


    def test_iterate(self):
        nlqt, x, u, r = self.setupNLQT()
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(nlqt.lqt)
        us0 = tf.convert_to_tensor(u)
        xs0 = tf.convert_to_tensor(x)

        cost0 = nlqt.cost(x[0], u)
        cost1 = cost0
        for i in range(5):
            u, x = nlqt.iterate(u, x)
            cost1 = nlqt.cost(x[0], u)
            print("iter cost1 = %f" % cost1)

        us = us0
        xs = xs0
        cost0 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
        cost2 = cost0
        for i in range(5):
            us, xs = nlqt_tf.nlqt_iterate_seq(us, xs, nlqt_tf_test_f, nlqt_tf_test_Fx, nlqt_tf_test_Fu, Hs, HT, rs, rT, Xs, XT,
                                      Us)
            cost2 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
            print("iter cost2 = %f" % cost2)

        us = us0
        xs = xs0
        cost0 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
        cost3 = cost0
        for i in range(5):
            us, xs = nlqt_tf.nlqt_iterate_par_1(us, xs, nlqt_tf_test_f, nlqt_tf_test_Fx, nlqt_tf_test_Fu, Hs, HT, rs, rT, Xs,
                                        XT, Us)
            cost3 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
            print("iter cost3 = %f" % cost3)

        us = us0
        xs = xs0
        cost0 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
        cost4 = cost0
        for i in range(5):
            us, xs = nlqt_tf.nlqt_iterate_par_2(us, xs, nlqt_tf_test_f, nlqt_tf_test_Fx, nlqt_tf_test_Fu, Hs, HT, rs, rT, Xs,
                                        XT, Us)
            cost4 = nlqt_tf.nlqt_cost(x[0], us, nlqt_tf_test_f, Hs, HT, rs, rT, Xs, XT, Us)
            print("iter cost4 = %f" % cost4)

        self.assertTrue(tf.math.abs(cost2 - cost1) < 1e-10)
        self.assertTrue(tf.math.abs(cost3 - cost1) < 1e-10)
        self.assertTrue(tf.math.abs(cost4 - cost1) < 1e-10)

