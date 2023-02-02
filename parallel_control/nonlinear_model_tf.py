"""
TensorFlow version of a nonlinear model for tracking position/orientation trajectory.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import parallel_control.nonlinear_model_np as nonlinear_model_np

import unittest

###########################################################################
# Nonlinear model functions
###########################################################################

def nonlinear_model_f(xs, us, dt=0.1):
    x_new = tf.stack([xs[:,0] + xs[:,3] * tf.cos(xs[:,2]) * dt,
                      xs[:,1] + xs[:,3] * tf.sin(xs[:,2]) * dt,
                      xs[:,2] + us[:,1] * dt,
                      xs[:,3] + us[:,0] * dt], axis=1)
    return x_new

def nonlinear_model_Fx(xs, us, dt=0.1):
    d_dx1 = tf.stack([tf.ones_like(xs[:,0]),
                      tf.zeros_like(xs[:,0]),
                      tf.zeros_like(xs[:,0]),
                      tf.zeros_like(xs[:,0])], axis=1)
    d_dx2 = tf.stack([tf.zeros_like(xs[:,0]),
                      tf.ones_like(xs[:,0]),
                      tf.zeros_like(xs[:,0]),
                      tf.zeros_like(xs[:,0])], axis=1)
    d_dx3 = tf.stack([-xs[:,3] * tf.sin(xs[:,2]) * dt,
                      xs[:,3] * tf.cos(xs[:,2]) * dt,
                      tf.ones_like(xs[:,0]),
                      tf.zeros_like(xs[:,0])], axis=1)
    d_dx4 = tf.stack([tf.cos(xs[:,2]) * dt,
                      tf.sin(xs[:,2]) * dt,
                      tf.zeros_like(xs[:,0]),
                      tf.ones_like(xs[:,0])], axis=1)
    return tf.stack([d_dx1, d_dx2, d_dx3, d_dx4], axis=2)

def nonlinear_model_Fu(xs, us, dt=0.1):
    d_du1 = tf.stack([tf.zeros_like(us[:,0]),
                      tf.zeros_like(us[:,0]),
                      tf.zeros_like(us[:,0]),
                      tf.ones_like(us[:,0]) * dt], axis=1)
    d_du2 = tf.stack([tf.zeros_like(us[:,0]),
                      tf.zeros_like(us[:,0]),
                      tf.ones_like(us[:, 0]) * dt,
                      tf.zeros_like(us[:,0])], axis=1)
    return tf.stack([d_du1, d_du2], axis=2)

