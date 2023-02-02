#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Tensorflow and Jax versions of (1d) nonlinear models.

@author: Simo Särkkä
"""

import tensorflow as tf
import jax
import jax.numpy as jnp
from jax.config import config

import parallel_control.cnonlin_models_np as cnonlin_models_np
import parallel_control.cnonlin_models_tf as cnonlin_models_tf
import unittest

##############################################################################
# Unit tests for SQP (TF version)
##############################################################################

class CNonLin_tf_UnitTest(unittest.TestCase):
    def test_linear_model(self):
        config.update("jax_enable_x64", True)

        f1, L1, LT1, T1, x_grid1, u_grid1 = cnonlin_models_np.linear_model(10,20)
        f2, L2, LT2, T2, x_grid2, u_grid2 = cnonlin_models_tf.linear_model(10,20)

        x = 1.2
        u = 0.3

        err = tf.math.abs(f1(x,u) - f2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(L1(x,u) - L2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(LT1(x) - LT2(x))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(T1 - T2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(x_grid1 - x_grid2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(u_grid1 - u_grid2)
        self.assertTrue(err < 1e-5)

    def test_velocity_model(self):
        config.update("jax_enable_x64", True)

        f1, L1, LT1, T1, x_grid1, u_grid1 = cnonlin_models_np.velocity_model(10,20)
        f2, L2, LT2, T2, x_grid2, u_grid2 = cnonlin_models_tf.velocity_model(10,20)

        x = 1.2
        u = 0.3

        err = tf.math.abs(f1(x,u) - f2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(L1(x,u) - L2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(LT1(x) - LT2(x))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(T1 - T2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(x_grid1 - x_grid2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(u_grid1 - u_grid2)
        self.assertTrue(err < 1e-5)

    def test_upwind_model(self):
        config.update("jax_enable_x64", True)

        f1, L1, LT1, T1, x_grid1, u_grid1 = cnonlin_models_np.upwind_model(10,20)
        f2, L2, LT2, T2, x_grid2, u_grid2 = cnonlin_models_tf.upwind_model(10,20)

        x = 1.2
        u = 0.3

        err = tf.math.abs(f1(x,u) - f2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(L1(x,u) - L2(x,u))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(LT1(x) - LT2(x))
        self.assertTrue(err < 1e-5)

        err = tf.math.abs(T1 - T2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(x_grid1 - x_grid2)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(u_grid1 - u_grid2)
        self.assertTrue(err < 1e-5)

    def test_upwind_model_block_V(self):
        config.update("jax_enable_x64", True)

        dtype = jnp.float64
        x_grid1 = jnp.linspace(0.0, 1.0, 10, dtype=dtype)
        inf_value1 = 100.0
        block_dt1 = 0.1

        V1 = cnonlin_models_np.upwind_model_block_V(inf_value1, block_dt1, x_grid1)

        dtype = tf.float64
        x_grid2 = tf.linspace(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype), 10)
        inf_value2 = tf.constant(100.0, dtype=dtype)
        block_dt2 = tf.constant(0.1, dtype=dtype)

        V2 = cnonlin_models_tf.upwind_model_block_V(inf_value2, block_dt2, x_grid2)

        err = tf.linalg.norm(V1 - V2)
        self.assertTrue(err < 1e-5)
