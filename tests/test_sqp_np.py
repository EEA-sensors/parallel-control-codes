#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Numpy/Jax versions sequential quadratic programming (with equality constraints).

@author: Simo Särkkä
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

import scipy.optimize

import parallel_control.sqp_np as sqp_np
import math
import unittest



##############################################################################
# Unit tests for SQP
##############################################################################

class SQP_np_UnitTest(unittest.TestCase):
    """Unit tests for SQP """

    def test_lqe_solve(self):
        config.update("jax_enable_x64", True)

        G = jnp.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 0.2],
                       [0.5, 0.2, 3.0]])

        c = jnp.array([0.1, 0.2, 0.3])
        A = jnp.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0]])
        b = jnp.array([0.1, 0.2])
        # A = jnp.array([[1.0, -1.0, 0.0]])
        # b = jnp.array([0.1])

        x1, lam = sqp_np.lqe_solve(G, c, A, b)
#        print(f'x1 = {x1}')

        fun_quad = lambda x: 0.5 * jnp.dot(x, G @ x) + jnp.dot(c, x)

        con_quad0 = lambda x: jnp.dot(A[0, :], x) - b[0]
        con_quad1 = lambda x: jnp.dot(A[1, :], x) - b[1]

        eq_cons = [{'type': 'eq', 'fun': con_quad0, 'jac': jax.jacfwd(con_quad0)},
                   {'type': 'eq', 'fun': con_quad1, 'jac': jax.jacfwd(con_quad1)}]

        x_init = jnp.array([0.0, 0.0, 0.0])
        res = scipy.optimize.minimize(fun_quad, x_init, method='SLSQP', jac=jax.jacfwd(fun_quad),
                                      constraints=eq_cons, options={'ftol': 1e-9, 'disp': True})

        x2 = res.x
#        print(f'x2 = {x2}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_local_1(self):
        config.update("jax_enable_x64", True)

        G = jnp.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 0.2],
                       [0.5, 0.2, 3.0]])

        c = jnp.array([0.1, 0.2, 0.3])
        A = jnp.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0]])
        b = jnp.array([0.1, 0.2])
#        A = jnp.array([[1.0, -1.0, 0.0]])
#        b = jnp.array([0.1])

        x1, lam = sqp_np.lqe_solve(G, c, A, b)
        print(f'x1 = {x1}')

        fun_quad = lambda x: 0.5 * jnp.dot(x, G @ x) + jnp.dot(c, x)
        con_quad = lambda x: A @ x - b

        x_init = jnp.array([0.0, 0.0, 0.0])
        x2, iter, crit = sqp_np.local_eq(fun_quad, con_quad, x_init)
        print(f'x2 = {x2}, iter = {iter}, crit={crit}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_local_2(self):
        config.update("jax_enable_x64", True)

        test_fun = lambda x: x[0] ** 4 - 2.0 * x[1] * x[0] ** 2 + x[1] ** 2 + x[0] ** 2 - 2.0 * x[0] + 5.0
        test_con = lambda x: jnp.array([-(x[0] + 0.25) ** 2 + 0.75 * x[1]])

        x0 = jnp.array([-1.0, 4.0])
        eq_cons = [{'type': 'eq', 'fun': test_con, 'jac': jax.jacfwd(test_con)}]

        res = scipy.optimize.minimize(test_fun, x0, method='SLSQP', jac=jax.jacfwd(test_fun),
                                      constraints=eq_cons, options={'ftol': 1e-9, 'disp': True})
        x1 = res.x
        print(f'x1 = {x1}')

        x2, iter, crit = sqp_np.local_eq(test_fun, test_con, x0, 100)
        print(f'x2 = {x2}, iter = {iter}, crit={crit}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_local_3(self):
        config.update("jax_enable_x64", True)

        test_fun = lambda x: x[0] ** 4 - 2.0 * x[1] * x[0] ** 2 + x[1] ** 2 + x[0] ** 2 - 2.0 * x[0] + 5.0
        test_con = lambda x: jnp.array([-(x[0] + 0.25) ** 2 + 0.75 * x[1]])

        x0 = jnp.array([-1.0, 4.0])

        x1, iter, crit = sqp_np.local_eq(test_fun, test_con, x0, 10)
        print(f'x1 = {x1}, iter = {iter}, crit={crit}')

        x2, iter, crit = sqp_np.local_eq_fast(test_fun, test_con, x0, 10, quiet=False)
        print(f'x2 = {x2}, iter = {iter}, crit={crit}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_bfgs_1(self):
        config.update("jax_enable_x64", True)

        G = jnp.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 0.2],
                       [0.5, 0.2, 3.0]])

        c = jnp.array([0.1, 0.2, 0.3])
        A = jnp.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0]])
        b = jnp.array([0.1, 0.2])
#        A = jnp.array([[1.0, -1.0, 0.0]])
#        b = jnp.array([0.1])

        x1, lam = sqp_np.lqe_solve(G, c, A, b)
        print(f'x1 = {x1}')

        fun_quad = lambda x: 0.5 * jnp.dot(x, G @ x) + jnp.dot(c, x)
        con_quad = lambda x: A @ x - b

        x_init = jnp.array([0.0, 0.0, 0.0])
        x2, iter, crit = sqp_np.bfgs_eq_bt(fun_quad, con_quad, x_init)
        print(f'x2 = {x2}, iter = {iter}, crit={crit}')

        x_init = jnp.array([0.0, 0.0, 0.0])
        x3, iter, crit = sqp_np.bfgs_eq_bt(fun_quad, con_quad, x_init, damped=False)
        print(f'x3 = {x3}, iter = {iter}, crit={crit}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

        err = jnp.linalg.norm(x1 - x3)
        print(err)
        self.assertTrue(err < 1e-5)

    def test_bfgs_2(self):
        config.update("jax_enable_x64", True)

        test_fun = lambda x: x[0] ** 4 - 2.0 * x[1] * x[0] ** 2 + x[1] ** 2 + x[0] ** 2 - 2.0 * x[0] + 5.0
        test_con = lambda x: jnp.array([-(x[0] + 0.25) ** 2 + 0.75 * x[1]])

        x0 = jnp.array([-1.0, 4.0], dtype=jnp.float64)
        print(x0.dtype)

        eq_cons = [{'type': 'eq', 'fun': test_con, 'jac': jax.jacfwd(test_con)}]

        res = scipy.optimize.minimize(test_fun, x0, method='SLSQP', jac=jax.jacfwd(test_fun),
                                      constraints=eq_cons, options={'ftol': 1e-9, 'disp': True})
        x1 = res.x
        print(f'x1 = {x1}')

        x2, iter, crit = sqp_np.bfgs_eq_bt(test_fun, test_con, x0, 100)
        print(f'x2 = {x2}, iter = {iter}, crit={crit}')

        x3, iter, crit = sqp_np.bfgs_eq_bt(test_fun, test_con, x0, 100, damped=False)
        print(f'x3 = {x3}, iter = {iter}, crit={crit}')

        err = jnp.linalg.norm(x1 - x2)
        print(err)
        self.assertTrue(err < 1e-5)

        err = jnp.linalg.norm(x1 - x3)
        print(err)
        self.assertTrue(err < 1e-5)
