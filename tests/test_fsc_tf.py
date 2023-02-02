"""
Unit tests for TensorFlow-based optimal finite state control (FSC).

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math

import unittest
import parallel_control.fsc_np as fsc_np
import parallel_control.fsc_tf as fsc_tf


class FSC_tf_UnitTest(unittest.TestCase):

    def setupFSC(self):
        xdim = 4
        udim = 3

        rng = np.random.default_rng(123)
        track = np.array([[0,0,2,5,5,5,1,1,0],
                          [0,1,2,2,2,1,2,2,1],
                          [0,1,2,2,2,1,2,2,1],
                          [0,0,0,0,0,5,5,5,5]])
        track = track + 0.01 * rng.uniform(0.0,1.0,size=track.shape) # To make solution unique

        # u = 0,1,2 = left,straight,right
        f = np.array([[0,0,1],
                      [0,1,2],
                      [1,2,3],
                      [2,3,3]], dtype=int)
        LT = np.array([0.9,1.0,1.1,1.2])
        x0 = 1

        T = track.shape[1]
        L = []
        u_cost = [1.0,0.0,1.0]
        for k in range(T):
            curr_L = np.zeros((xdim,udim))
            for x in range(xdim):
                for u in range(udim):
                    curr_L[x,u] = track[x,k] + u_cost[u]
            L.append(curr_L)

        fsc = fsc_np.FSC.checkAndExpand(f, L, LT)

        return fsc, x0

    def setupFSC_tf_inputs(self, fsc):
         return fsc_tf.fsc_np_to_tf(fsc)

    def test_sequential(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()
        min_u_list1, min_x_list1 = fsc.seqForwardPass(x0,u_list1)
        min_cost1 = V_list1[0][x0]

        fsc, x0 = self.setupFSC()
        fs, Ls, LT = self.setupFSC_tf_inputs(fsc)

        us, Vs = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
        min_cost = Vs[0,x0]

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs.dtype) - Vs))
        self.assertTrue(err < 1e-6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list1, dtype=us.dtype) - us))
        self.assertTrue(err == 0)

        min_xs, min_us = fsc_tf.fsc_seq_forwardpass(tf.constant(x0, dtype=fs.dtype), fs, us)

        print(min_u_list1)
        print(min_x_list1)
        print(min_cost1)
        print(min_us)
        print(min_xs)
        print(min_cost)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_x_list1, dtype=min_xs.dtype) - min_xs))
        self.assertTrue(err == 0)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_u_list1, dtype=min_us.dtype) - min_us))
        self.assertTrue(err == 0)


    def test_parback(self):
        fsc, x0 = self.setupFSC()
        fs, Ls, LT = self.setupFSC_tf_inputs(fsc)

        u_list1, V_list1 = fsc.parBackwardPass()

        us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs.dtype) - Vs))
        self.assertTrue(err < 1e-6)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(u_list1, dtype=us.dtype) - us))
        self.assertTrue(err == 0)


    def test_parfwd_1(self):
        fsc, x0 = self.setupFSC()
        fs, Ls, LT = self.setupFSC_tf_inputs(fsc)

        u_list1, V_list1 = fsc.parBackwardPass()
        min_u_list1, min_x_list1 = fsc.parForwardPass(x0, u_list1)
        min_cost1 = V_list1[0][x0]

        us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT)
        min_us, min_xs = fsc_tf.fsc_par_forwardpass(x0, fs, us)
        min_cost = Vs[0,x0]

        print(min_u_list1)
        print(min_x_list1)
        print(min_cost1)
        print(min_us)
        print(min_xs)
        print(min_cost)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_x_list1, dtype=min_xs.dtype) - min_xs))
        self.assertTrue(err == 0)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_u_list1, dtype=min_us.dtype) - min_us))
        self.assertTrue(err == 0)


    def test_parfwd_2(self):
        fsc, x0 = self.setupFSC()
        fs, Ls, LT = self.setupFSC_tf_inputs(fsc)

        u_list1, V_list1 = fsc.parBackwardPass()
        elems = fsc.parFwdBwdPass_init(x0)

        min_u_list1, min_x_list1 = fsc.parFwdBwdPass(x0, u_list1, V_list1)
        min_cost1 = V_list1[0][x0]

        us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT)

        min_us, min_xs = fsc_tf.fsc_par_fwdbwdpass(x0, fs, Ls, us, Vs)
        min_cost = Vs[0,x0]

        print(min_u_list1)
        print(min_x_list1)
        print(min_cost1)
        print(min_us)
        print(min_xs)
        print(min_cost)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_x_list1, dtype=min_xs.dtype) - min_xs))
        self.assertTrue(err == 0)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(min_u_list1, dtype=min_us.dtype) - min_us))
        self.assertTrue(err == 0)

