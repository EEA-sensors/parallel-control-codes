#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for TensorFlow versions sequential quadratic programming (with equality constraints).

@author: Simo Särkkä
"""

import tensorflow as tf
import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

import parallel_control.sqp_np as sqp_np
import parallel_control.sqp_tf as sqp_tf
import math
import unittest



##############################################################################
# Unit tests for SQP (TF version)
##############################################################################

class SQP_tf_UnitTest(unittest.TestCase):
    """Unit tests for SQP """

    def test_lqe_solve_1(self):
        config.update("jax_enable_x64", True)

        G = jnp.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 0.2],
                       [0.5, 0.2, 3.0]])

        c = jnp.array([0.1, 0.2, 0.3])
        A = jnp.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0]])
        b = jnp.array([0.1, 0.2])

        x1, lam1 = sqp_np.lqe_solve(G, c, A, b)
        print(f'x1 = {x1}, lam1={lam1}')

        x2, lam2 = sqp_tf.lqe_solve(G, c, A, b)
        print(f'x2 = {x2}, lam2={lam2}')

        err = tf.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(lam1 - lam2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_lqe_solve_2(self):
        config.update("jax_enable_x64", True)

        G = jnp.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 0.2],
                       [0.5, 0.2, 3.0]])

        c = jnp.array([0.1, 0.2, 0.3])
        A = jnp.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0]])
        b = jnp.array([0.1, 0.2])

        x1, lam1 = sqp_np.lqe_solve(G, c, A, b)
        print(f'x1 = {x1}, lam1={lam1}')

        G1 = tf.expand_dims(G, axis=0)
        c1 = tf.expand_dims(c, axis=0)
        A1 = tf.expand_dims(A, axis=0)
        b1 = tf.expand_dims(b, axis=0)
        x2, lam2 = sqp_tf.lqe_solve(G1, c1, A1, b1)
        print(f'x2 = {x2}, lam2={lam2}')

        err = tf.linalg.norm(x1 - x2[0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(lam1 - lam2[0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

        G2 = tf.expand_dims(G1, axis=0)
        G2 = tf.concat((G2,G2), axis=0)
        c2 = tf.expand_dims(c1, axis=0)
        c2 = tf.concat((c2,c2), axis=0)
        A2 = tf.expand_dims(A1, axis=0)
        A2 = tf.concat((A2,A2), axis=0)
        b2 = tf.expand_dims(b1, axis=0)
        b2 = tf.concat((b2,b2), axis=0)
        x3, lam3 = sqp_tf.lqe_solve(G2, c2, A2, b2)
        print(f'x3 = {x3}, lam3={lam3}')

        err = tf.linalg.norm(x1 - x3[0, 0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(lam1 - lam3[0, 0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(x1 - x3[1, 0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(lam1 - lam3[1, 0, ...])
        print(err)
        self.assertTrue(err < 1e-5)

    def test_grad_1(self):
        config.update("jax_enable_x64", True)

        jax_f = lambda x: x[0]**2 + jnp.sin(x[1]) * x[0]
        tf_f = lambda x, p: x[..., 0]**2 + tf.sin(x[..., 1]) * x[..., 0]

        jax_grad = jax.jacfwd(jax_f)

        jax_x = jnp.array([1.0, 2.0])

        value1 = jax_f(jax_x)
        grad1 = jax_grad(jax_x)
#        print(value1)
#        print(grad1)

        tf_x = tf.constant([1.0, 2.0], dtype=tf.float64)
        value2, grad2 = sqp_tf.fun_value_grad(tf_f, tf_x, 0)
#        print(value2)
#        print(grad2)

        err = tf.linalg.norm(value1 - value2)
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(grad1 - grad2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_grad_2(self):
        config.update("jax_enable_x64", True)

        jax_f = lambda x: x[0]**2 + jnp.sin(x[1]) * x[0]
        tf_f = lambda x, p: x[..., 0]**2 + tf.sin(x[..., 1]) * x[..., 0]

        jax_grad = jax.jacfwd(jax_f)

        jax_xa = jnp.array([1.0, 2.0])
        jax_xb = jnp.array([1.2, 2.2])

        value1a = jax_f(jax_xa)
        grad1a = jax_grad(jax_xa)
        value1b = jax_f(jax_xb)
        grad1b = jax_grad(jax_xb)
#        print(value1a)
#        print(grad1a)
#        print(value1b)
#        print(grad1b)

        tf_x = tf.constant([[1.0, 2.0], [1.2, 2.2]], dtype=tf.float64)
        value2, grad2 = sqp_tf.fun_value_grad(tf_f, tf_x, 0, batch=True)
#        print(value2)
#        print(grad2)

        err = tf.linalg.norm(value1a - value2[0])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(grad1a - grad2[0])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(grad1b - grad2[1])
        print(err)
        self.assertTrue(err < 1e-5)

    def test_jac_1(self):
        config.update("jax_enable_x64", True)

        jax_f = lambda x: jnp.array([x[0] ** 2 + jnp.sin(x[1]) * x[0], x[1]**2])
        tf_f = lambda x, p: tf.stack([x[..., 0] ** 2 + tf.sin(x[..., 1]) * x[..., 0], x[..., 1] ** 2])

        jax_jac = jax.jacfwd(jax_f)

        jax_x = jnp.array([1.0, 2.0])

        value1 = jax_f(jax_x)
        jac1 = jax_jac(jax_x)
        print(value1)
        print(jac1)

        tf_x = tf.constant([1.0, 2.0], dtype=tf.float64)
        value2, jac2 = sqp_tf.con_value_jac(tf_f, tf_x, 0)
        print(value2)
        print(jac2)

        err = tf.linalg.norm(value1 - value2)
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(jac1 - jac2)
        print(err)
        self.assertTrue(err < 1e-5)


    def test_jac_2(self):
        config.update("jax_enable_x64", True)

        jax_f = lambda x: jnp.array([x[0] ** 2 + jnp.sin(x[1]) * x[0], jnp.cos(x[0]) + x[1]**2])
        tf_f = lambda x, p: tf.stack([x[..., 0] ** 2 + tf.sin(x[..., 1]) * x[..., 0], tf.cos(x[..., 0]) + x[..., 1] ** 2], axis=-1)

        jax_jac = jax.jacfwd(jax_f)

        jax_xa = jnp.array([1.0, 2.0])
        jax_xb = jnp.array([1.2, 2.2])

        value1a = jax_f(jax_xa)
        jac1a = jax_jac(jax_xa)
        value1b = jax_f(jax_xb)
        jac1b = jax_jac(jax_xb)
#        print(value1a)
#        print(jac1a)
#        print(value1b)
#        print(jac1b)

        tf_x = tf.constant([[1.0, 2.0], [1.2, 2.2]], dtype=tf.float64)
        value2, jac2 = sqp_tf.con_value_jac(tf_f, tf_x, 0, batch=True)
#        print(value2[0])
#        print(jac2[0])
#        print(value2[1])
#        print(jac2[1])

        err = tf.linalg.norm(value1a - value2[0])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(value1b - value2[1])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(jac1a - jac2[0])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(jac1b - jac2[1])
        print(err)
        self.assertTrue(err < 1e-5)


    def test_hess_1(self):
        config.update("jax_enable_x64", True)

        jax_con = lambda x: jnp.array([x[0] ** 2 + jnp.sin(x[1]) * x[0], jnp.cos(x[0]) + x[1]**2])
        tf_con = lambda x, p: tf.stack([x[..., 0] ** 2 + tf.sin(x[..., 1]) * x[..., 0], tf.cos(x[..., 0]) + x[..., 1] ** 2], axis=-1)

        jax_fun = lambda x: x[0]**2 + jnp.sin(x[1]) * x[0]
        tf_fun = lambda x, p: x[..., 0]**2 + tf.sin(x[..., 1]) * x[..., 0]

        lam = jnp.array([0.1, 0.2], dtype=jnp.float64)
        jax_f = lambda x: jax_fun(x) - jnp.dot(lam, jax_con(x))

        jax_grad = jax.jacfwd(jax_f)
        jax_hess = jax.jacfwd(jax_grad)

        jax_x = jnp.array([1.0, 2.0])

        value1 = jax_f(jax_x)
        grad1 = jax_grad(jax_x)
        hess1 = jax_hess(jax_x)

#        print(value1)
#        print(grad1)
#        print(hess1)

        tf_x = tf.constant([1.0, 2.0], dtype=tf.float64)
        lam_tf = tf.constant([0.1, 0.2], dtype=tf.float64)
        value2, grad2, hess2 = sqp_tf.lag_value_grad_hess(tf_fun, tf_con, tf_x, lam_tf, 0)
#        print(value2)
#        print(grad2)
#        print(hess2)

        err = tf.linalg.norm(hess1 - hess2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_hess_2(self):
        config.update("jax_enable_x64", True)

        jax_con = lambda x: jnp.array([x[0] ** 2 + jnp.sin(x[1]) * x[0], jnp.cos(x[0]) + x[1]**2])
        tf_con = lambda x, p: tf.stack([x[..., 0] ** 2 + tf.sin(x[..., 1]) * x[..., 0], tf.cos(x[..., 0]) + x[..., 1] ** 2], axis=-1)

        jax_fun = lambda x: x[0]**2 + jnp.sin(x[1]) * x[0]
        tf_fun = lambda x, p: x[..., 0]**2 + tf.sin(x[..., 1]) * x[..., 0]

        lam = jnp.array([0.1, 0.2], dtype=jnp.float64)
        jax_f = lambda x: jax_fun(x) - jnp.dot(lam, jax_con(x))

        jax_grad = jax.jacfwd(jax_f)
        jax_hess = jax.jacfwd(jax_grad)

        jax_xa = jnp.array([1.0, 2.0])
        jax_xb = jnp.array([1.2, 2.2])

        value1a = jax_f(jax_xa)
        hess1a = jax_hess(jax_xa)
        value1b = jax_f(jax_xb)
        hess1b = jax_hess(jax_xb)
#        print(value1a)
#        print(grad1a)
#        print(value1b)
#        print(grad1b)
#        print(hess1a)
#        print(hess1b)

        tf_x = tf.constant([[1.0, 2.0], [1.2, 2.2]], dtype=tf.float64)
        lam_tf = tf.constant([[0.1, 0.2],[0.1, 0.2]], dtype=tf.float64)
        value2, grad2, hess2 = sqp_tf.lag_value_grad_hess(tf_fun, tf_con, tf_x, lam_tf, 0, batch=True)
#        print(value2)
#        print(grad2)
#        print(hess2)

        err = tf.linalg.norm(hess1a - hess2[0])
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.linalg.norm(hess1b - hess2[1])
        print(err)
        self.assertTrue(err < 1e-5)

    def test_local_1(self):
        config.update("jax_enable_x64", True)

        test_fun = lambda x: x[0] ** 4 - 2.0 * x[1] * x[0] ** 2 + x[1] ** 2 + x[0] ** 2 - 2.0 * x[0] + 5.0
        test_con = lambda x: jnp.array([-(x[0] + 0.25) ** 2 + 0.75 * x[1]])

        dp = lambda v: tf.constant(v, dtype=tf.float64)
        test_fun_tf = lambda x, p: x[..., 0] ** 4 - dp(2.0) * x[..., 1] * x[..., 0] ** 2 + x[..., 1] ** 2 + x[..., 0] ** 2 - dp(2.0) * x[..., 0] + dp(5.0)
        test_con_tf = lambda x, p: tf.stack([-(x[..., 0] + dp(0.25)) ** 2 + dp(0.75) * x[..., 1]], axis=-1)


        x0 = jnp.array([-1.0, 4.0])

        x1, iter1, crit1 = sqp_np.local_eq_fast(test_fun, test_con, x0, 3, quiet=False)
        print(f'x1 = {x1}, crit={crit1}')

        x0_tf = tf.constant(x0, dtype=tf.float64)
        xs2, lams2, crits2 = sqp_tf.local_eq_fast(test_fun_tf, test_con_tf, x0, 0, 3)
        print(xs2)

        x2 = xs2[-1]
        crit2 = crits2[-1]
        print(f'x2 = {x2}, crit2={crit2}')

#        err = jnp.linalg.norm(x1 - x2)
#        print(err)
#        self.assertTrue(err < 1e-5)

    def test_local_2(self):
        config.update("jax_enable_x64", True)

        # TODO: Test with batched parameters and everything

