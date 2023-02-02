#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow routines for ordinary and partial differential equations

@author: Simo Särkkä
"""

import tensorflow as tf

#mm = tf.linalg.matmul
#mv = tf.linalg.matvec

@tf.function(reduce_retracing=True)
def rk4(f, dt, x, t=0.0, param=0.0):
    dx1 = f(x, t, param) * dt
    dx2 = f(x + 0.5 * dx1, t + 0.5 * dt, param) * dt
    dx3 = f(x + 0.5 * dx2, t + 0.5 * dt, param) * dt
    dx4 = f(x + dx3, t + dt, param) * dt

    x = x + (1.0 / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)

    return x

