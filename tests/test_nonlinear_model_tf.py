#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for TensorFlow version of a nonlinear model for tracking position/orientation trajectory.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import parallel_control.nonlinear_model_np as nonlinear_model_np
import parallel_control.nonlinear_model_tf as nonlinear_model_tf
import unittest

class NonlinearModel_tf_UnitTest(unittest.TestCase):
    """Unit tests for TF nonlinear model"""

    def test_func(self):
        rob = nonlinear_model_np.NonlinearModel()

        x_list = [np.array([0.1,0.2,0.3,0.4]), np.array([0.4,0.3,0.2,0.1])]
        u_list = [np.array([0.4,0.5]), np.array([0.2,0.3])]

        xs = tf.convert_to_tensor(x_list)
        us = tf.convert_to_tensor(u_list)

        x_new_list = [rob.f(x,u) for x,u in zip(x_list,u_list)]
        x_new = nonlinear_model_tf.nonlinear_model_f(xs, us)

        self.assertTrue(tf.reduce_max(tf.abs(tf.convert_to_tensor(x_new_list) - x_new)) < 1e-10)

    def test_der(self):
        rob = nonlinear_model_np.NonlinearModel()

        x_list = [np.array([0.1,0.2,0.3,0.4]), np.array([0.4,0.3,0.2,0.1])]
        u_list = [np.array([0.4,0.5]), np.array([0.2,0.3])]

        xs = tf.convert_to_tensor(x_list)
        us = tf.convert_to_tensor(u_list)

        Fx_list = [rob.Fx(x,u) for x,u in zip(x_list,u_list)]
        Fxs = nonlinear_model_tf.nonlinear_model_Fx(xs, us)

        Fu_list = [rob.Fu(x,u) for x,u in zip(x_list,u_list)]
        Fus = nonlinear_model_tf.nonlinear_model_Fu(xs, us)

        self.assertTrue(tf.reduce_max(tf.abs(tf.convert_to_tensor(Fx_list) - Fxs)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.abs(tf.convert_to_tensor(Fu_list) - Fus)) < 1e-10)

