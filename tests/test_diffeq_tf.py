#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for TensorFlow differential equation routines.

@author: Simo Särkkä
"""

import unittest
import numpy as np
import tensorflow as tf
import parallel_control.diffeq_tf as diffeq_tf

mm = tf.linalg.matmul
mv = tf.linalg.matvec

class DiffEq_tf_UnitTest(unittest.TestCase):
    def test_rk4_1(self):

        # dx/dt = -c x, x(0) = x0
        #  x(t) = x0 exp(-c t)
        c  = 0.5
        x0 = 0.7
        f  = lambda x, t, p: -c * x

        dt = 0.1
        x1 = diffeq_tf.rk4(f, dt, x0)
        x2 = x0 * tf.exp(-c * dt)
        self.assertTrue(tf.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1)
        x2 = x0 * tf.exp(-c * 1.0)

        self.assertTrue(tf.math.abs(x2 - x1) < 1e-5)

    def test_rk4_2(self):

        # dx/dt = -cos(t) x, x(0) = x0
        #  x(t) = x0 exp(-int_0^t cos(t) dt) = x0 exp(-sin(t))

        x0 = 0.7
        f = lambda x, t, p: -tf.cos(t) * x

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1, k * dt)
        x2 = x0 * tf.exp(-tf.sin(1.0))

        self.assertTrue(tf.math.abs(x2 - x1) < 1e-5)

    def test_rk4_3(self):
        x0 = tf.constant(np.array([1,2,3]), dtype=tf.float64)
        F  = tf.constant(np.array([[0,1,0],[0,0,1],[-0.1,-0.2,-0.3]]), dtype=tf.float64)

        f = lambda x, t, p: mv(F, x)
        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1)
        x2 = mv(tf.linalg.expm(F * 1.0), x0)

        self.assertTrue(tf.linalg.norm(x2 - x1) < 1e-5)

    def test_rk4_4(self):
        # dx/dt = -c x, x(T) = x0
        #  x(t) = x0 exp(-c t + c T)
        c  = 0.5
        x0 = 0.7
        f  = lambda x, t, p: -c * x

        dt = 0.1
        x1 = diffeq_tf.rk4(f, -dt, x0)
        x2 = x0 * tf.exp(-c * 0 + c * dt)

        self.assertTrue(tf.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, -dt, x1)
        x2 = x0 * tf.exp(-c * 0 + c * 1.0)

        self.assertTrue(tf.math.abs(x2 - x1) < 1e-5)

    def test_rk4_5(self):

        # dx/dt = -c x, x(0) = x0
        #  x(t) = x0 exp(-c t)
        f  = lambda x, t, c: -c * x
        c  = 0.5
        x0 = 0.7

        dt = 0.1
        x1 = diffeq_tf.rk4(f, dt, x0, param=c)
        x2 = x0 * tf.exp(-c * dt)
        self.assertTrue(tf.abs(x2 - x1) < 1e-5)

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1, param=c)
        x2 = x0 * tf.exp(-c * 1.0)

        self.assertTrue(tf.math.abs(x2 - x1) < 1e-5)

    def test_rk4_6(self):

        # dx/dt = -c cos(t) x, x(0) = x0
        #  x(t) = x0 exp(-int_0^t c cos(t) dt) = x0 exp(-c sin(t))

        x0 = 0.7
        f = lambda x, t, c: -c * tf.cos(t) * x
        c = 2

        steps = 10
        dt = 1.0 / steps
        x1 = x0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1, k * dt, c)
        x2 = x0 * tf.exp(-c*tf.sin(1.0))

        self.assertTrue(tf.math.abs(x2 - x1) < 1e-5)

    def test_rk4_7(self):

        # Test batching
        # dx/dt = -c cos(t) x, x(t0) = x0
        #  x(t) = x0 exp(-int_t0^t c cos(t) dt) = x0 exp(-c (sin(t) - sin(t0)))

        f = lambda x, t, c: -c * tf.cos(t) * x
        c = 2.0

        steps = 20
        dt = 1.0 / steps
        x0 = tf.constant([0.5,0.7,0.8], dtype=tf.float64)
        x1 = x0
        t0 = tf.constant([0.0,1.0,2.0], dtype=tf.float64)
        t1 = t0
        for k in range(steps):
            x1 = diffeq_tf.rk4(f, dt, x1, t1, c)
            t1 = t1 + dt
        x2 = x0 * tf.exp(-c * (tf.sin(t1) - tf.sin(t0)))

        print(x1)
        print(x2)

        self.assertTrue(tf.math.reduce_max(tf.math.abs(x2 - x1)) < 1e-5)

