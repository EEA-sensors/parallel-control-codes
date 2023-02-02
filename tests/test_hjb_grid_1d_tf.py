#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for TensorFlow versions of routines for solving continuous-time nonlinear 1d problems.

@author: Simo Särkkä
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.config import config

import parallel_control.hjb_grid_1d_np as hjb_grid_1d_np
import parallel_control.hjb_grid_1d_tf as hjb_grid_1d_tf
import parallel_control.cnonlin_models_np as cnonlin_models_np
import parallel_control.cnonlin_models_tf as cnonlin_models_tf
import parallel_control.sqp_np as sqp_np
import parallel_control.sqp_tf as sqp_tf

import time
import math
import unittest

##############################################################################
# Unit tests for HJB on grid
##############################################################################

class HJB_Grid_1d_tf_UnitTest(unittest.TestCase):
    def test_fun_and_con_1_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        a_vec_np = jnp.array([0.1, 0.2, 0.3])
        b_vec_np = jnp.array([0.3, 0.2, 0.1])

        x0 = 0.3
        xf = 0.4

        fv1, cv1 = hjb_grid_1d_np.fun_and_con_1(f, L, T, x0, xf, a_vec_np, b_vec_np)

        print(f"fv1 = {fv1}, cv1 = {cv1}")

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        a_vec_tf = tf.constant(a_vec_np, dtype=tf.float64)
        b_vec_tf = tf.constant(b_vec_np, dtype=tf.float64)

        x0_tf = tf.constant(x0, dtype=tf.float64)
        xf_tf = tf.constant(xf, dtype=tf.float64)

        x0_tf = tf.expand_dims(x0_tf, 0)
        xf_tf = tf.expand_dims(xf_tf, 0)
        a_vec_tf = tf.expand_dims(a_vec_tf, 0)
        b_vec_tf = tf.expand_dims(b_vec_tf, 0)

        fv2, cv2 = hjb_grid_1d_tf.fun_and_con_1(f, L, T, x0_tf, xf_tf, a_vec_tf, b_vec_tf)

        print(f"fv2 = {fv2}, cv2 = {cv2}")

        err = tf.reduce_max(tf.math.abs(fv1 - fv2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(cv1 - cv2))
        self.assertTrue(err < 1e-5)

    def test_fun_and_con_1_2(self):
        config.update("jax_enable_x64", True)

        steps = 5
        theta0 = jnp.zeros((2 * steps,), dtype=jnp.float64)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        x0 = 0.3
        xf = 0.4
        max_iter = 4

        theta1, iter1, crit1 = sqp_np.local_eq_fast(lambda theta: hjb_grid_1d_np.fun_and_con_1(f, L, T, x0, xf, theta[0::2], theta[1::2])[0],
                                                    lambda theta: hjb_grid_1d_np.fun_and_con_1(f, L, T, x0, xf, theta[0::2], theta[1::2])[1],
                                                    theta0, max_iter, quiet=False)
        print(f"theta1={theta1}, crit1={crit1}")

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        x0_tf = tf.constant(x0, dtype=tf.float64)
        xf_tf = tf.constant(xf, dtype=tf.float64)
        theta0_tf = tf.constant(theta0, dtype=tf.float64)

        x0_tf = tf.expand_dims(x0_tf, 0)
        xf_tf = tf.expand_dims(xf_tf, 0)
        theta0_tf = tf.expand_dims(theta0_tf, 0)

        @tf.function
        def fun(theta, par):
            x0 = par[..., 0]
            xf = par[..., 1]
            a_vec = theta[..., 0::2]
            b_vec = theta[..., 1::2]
            fun_v, _ = hjb_grid_1d_tf.fun_and_con_1(f, L, T, x0, xf, a_vec, b_vec)
            return fun_v

        @tf.function
        def con(theta, par):
            x0 = par[..., 0]
            xf = par[..., 1]
            a_vec = theta[..., 0::2]
            b_vec = theta[..., 1::2]
            _, con_v = hjb_grid_1d_tf.fun_and_con_1(f, L, T, x0, xf, a_vec, b_vec)
            return con_v

        tic = time.time()
        thetas2, lams2, crits2 = sqp_tf.local_eq_fast(fun, con, theta0_tf, tf.stack([x0_tf, xf_tf], axis=-1), max_iter, batch=True)
        toc = time.time()
        run1 = toc - tic

        tic = time.time()
        thetas2, lams2, crits2 = sqp_tf.local_eq_fast(fun, con, theta0_tf, tf.stack([x0_tf, xf_tf], axis=-1), max_iter, batch=True)
        toc = time.time()
        run2 = toc - tic

        print(f"run1 = {run1}, run2={run2}")

        theta2 = thetas2[-1, ...]
        print(theta2)

        err = tf.reduce_max(tf.math.abs(theta1 - theta2))
        self.assertTrue(err < 1e-5)

    def test_V_eval_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model(10, 10)

        max_iter = 4
        steps = 10
        V1 = hjb_grid_1d_np.V_eval_1(f, L, T, steps, x_grid, x_grid, max_iter)

        print(V1)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model(10, 10)
        V2 = hjb_grid_1d_tf.V_eval_1(f, L, T, steps, x_grid, x_grid, max_iter)

        print(V2)

        err = tf.reduce_max(tf.math.abs(V1 - V2))
        self.assertTrue(err < 1e-5)

    def test_fun_and_con_2_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        theta_np = jnp.array([0.2, 0.1, 0.2, 0.3])

        x0 = 0.3
        xf = 0.4

        fv1, cv1 = hjb_grid_1d_np.fun_and_con_2(f, L, T, x0, xf, theta_np)

        print(f"fv1 = {fv1}, cv1 = {cv1}")

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        theta_tf = tf.constant(theta_np, dtype=tf.float64)

        x0_tf = tf.constant(x0, dtype=tf.float64)
        xf_tf = tf.constant(xf, dtype=tf.float64)

        x0_tf = tf.expand_dims(x0_tf, 0)
        xf_tf = tf.expand_dims(xf_tf, 0)
        a_vec_tf = tf.expand_dims(theta_tf, 0)

        fv2, cv2 = hjb_grid_1d_tf.fun_and_con_2(f, L, T, x0_tf, xf_tf, theta_tf)

        print(f"fv2 = {fv2}, cv2 = {cv2}")

        err = tf.reduce_max(tf.math.abs(fv1 - fv2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(cv1 - cv2))
        self.assertTrue(err < 1e-5)

    def test_fun_and_con_2_2(self):
        config.update("jax_enable_x64", True)

        steps = 5
        theta0 = jnp.zeros((1 + steps,), dtype=jnp.float64)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        x0 = 0.3
        xf = 0.4
        max_iter = 4

        theta1, iter1, crit1 = sqp_np.local_eq_fast(lambda theta: hjb_grid_1d_np.fun_and_con_2(f, L, T, x0, xf, theta)[0],
                                                    lambda theta: hjb_grid_1d_np.fun_and_con_2(f, L, T, x0, xf, theta)[1],
                                                    theta0, max_iter, quiet=False)
        print(f"theta1={theta1}, crit1={crit1}")

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        x0_tf = tf.constant(x0, dtype=tf.float64)
        xf_tf = tf.constant(xf, dtype=tf.float64)
        theta0_tf = tf.constant(theta0, dtype=tf.float64)

        x0_tf = tf.expand_dims(x0_tf, 0)
        xf_tf = tf.expand_dims(xf_tf, 0)
        theta0_tf = tf.expand_dims(theta0_tf, 0)

        @tf.function
        def fun(theta, par):
            x0 = par[..., 0]
            xf = par[..., 1]
            fun_v, _ = hjb_grid_1d_tf.fun_and_con_2(f, L, T, x0, xf, theta)
            return fun_v

        @tf.function
        def con(theta, par):
            x0 = par[..., 0]
            xf = par[..., 1]
            _, con_v = hjb_grid_1d_tf.fun_and_con_2(f, L, T, x0, xf, theta)
            return con_v

        tic = time.time()
        thetas2, lams2, crits2 = sqp_tf.local_eq_fast(fun, con, theta0_tf, tf.stack([x0_tf, xf_tf], axis=-1), max_iter, batch=True)
        toc = time.time()
        run1 = toc - tic

        tic = time.time()
        thetas2, lams2, crits2 = sqp_tf.local_eq_fast(fun, con, theta0_tf, tf.stack([x0_tf, xf_tf], axis=-1), max_iter, batch=True)
        toc = time.time()
        run2 = toc - tic

        print(f"run1 = {run1}, run2={run2}")

        theta2 = thetas2[-1, ...]
        print(theta2)

        err = tf.reduce_max(tf.math.abs(theta1 - theta2))
        self.assertTrue(err < 1e-5)

    def test_V_eval_2(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model(10, 10)

        max_iter = 4
        steps = 10
        V1 = hjb_grid_1d_np.V_eval_2(f, L, T, steps, x_grid, x_grid, max_iter)

        print(V1)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model(10, 10)

        tic = time.time()
        V2 = hjb_grid_1d_tf.V_eval_2(f, L, T, steps, x_grid, x_grid, max_iter)
        toc = time.time()
        run1 = toc - tic

        tic = time.time()
        V2 = hjb_grid_1d_tf.V_eval_2(f, L, T, steps, x_grid, x_grid, max_iter)
        toc = time.time()
        run2 = toc - tic

        print(f"run1 = {run1}, run2={run2}")
        print(V2)

        err = tf.reduce_max(tf.math.abs(V1 - V2))
        self.assertTrue(err < 1e-5)

    def test_combine_1(self):
        N = 5
        Va = np.random.normal(0.0, 1.0, (N, N))
        Vb = np.random.normal(0.0, 1.0, (N, N))

        V0 = hjb_grid_1d_np.combine_V(Va, Vb)

        Va_tf = tf.expand_dims(tf.constant(Va, dtype=tf.float64), 0)
        Vb_tf = tf.expand_dims(tf.constant(Vb, dtype=tf.float64), 0)
        V1 = hjb_grid_1d_tf.combine_V(Va_tf, Vb_tf)
        V2 = hjb_grid_1d_tf.combine_V2(Va_tf, Vb_tf)

        print(V1)
        print(V2)
        err = tf.reduce_max(tf.math.abs(V1 - V0))
        self.assertTrue(err < 1e-5)
        err = tf.reduce_max(tf.math.abs(V2 - V0))
        self.assertTrue(err < 1e-5)

    def test_combine_2(self):
        M = 2
        N = 5
        Va = np.random.normal(0.0, 1.0, (M, N, N))
        Vb = np.random.normal(0.0, 1.0, (M, N, N))

        Va_tf = tf.constant(Va, dtype=tf.float64)
        Vb_tf = tf.constant(Vb, dtype=tf.float64)
        V1 = hjb_grid_1d_tf.combine_V(Va_tf, Vb_tf)
        V2 = hjb_grid_1d_tf.combine_V2(Va_tf, Vb_tf)

        print(V1)
        print(V2)
        err = tf.reduce_max(tf.math.abs(V2 - V1))
        self.assertTrue(err < 1e-5)

    def test_combine_3(self):
        N = 5
        Va = np.random.normal(0.0, 1.0, (N, N))
        Vb = np.random.normal(0.0, 1.0, (N, N))

        V0 = hjb_grid_1d_np.combine_V_interp(Va, Vb)

        Va_tf = tf.expand_dims(tf.constant(Va, dtype=tf.float64), 0)
        Vb_tf = tf.expand_dims(tf.constant(Vb, dtype=tf.float64), 0)
        V1 = hjb_grid_1d_tf.combine_V_interp(Va_tf, Vb_tf)

        print(V0)
        print(V1)
        err = tf.reduce_max(tf.math.abs(V1 - V0))
        self.assertTrue(err < 1e-5)

    def test_combine_4(self):
        M = 2
        N = 5
        Va = np.random.normal(0.0, 1.0, (M, N, N))
        Vb = np.random.normal(0.0, 1.0, (M, N, N))

        V0 = np.zeros_like(Va)

        for i in range(M):
            V0[i,:,:] = hjb_grid_1d_np.combine_V_interp(Va[i,:,:], Vb[i,:,:])

        Va_tf = tf.constant(Va, dtype=tf.float64)
        Vb_tf = tf.constant(Vb, dtype=tf.float64)
        V1 = hjb_grid_1d_tf.combine_V_interp(Va_tf, Vb_tf)

        print(V0)
        print(V1)
        err = tf.reduce_max(tf.math.abs(V1 - V0))
        self.assertTrue(err < 1e-5)


    def test_symfd_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        blocks = 10
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_symfd(u_grid)
        Vs1 = jnp.array(V_list)

        # This is inaccurate, but no can do
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD1')
        plt.show()

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        V_fds = hjb_grid_1d_tf.seq_bw_pass_symfd(f, L, LT, T, blocks, steps, x_grid, u_grid)
        Vs2 = V_fds.numpy()
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD2')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :])
        plt.show()

        err = jnp.max(jnp.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-5)


    def test_upwind_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        blocks = 10
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_upwind(u_grid)
        Vs1 = jnp.array(V_list)

        # This is inaccurate, but no can do
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD1')
        plt.show()

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        V_fds = hjb_grid_1d_tf.seq_bw_pass_upwind(f, L, LT, T, blocks, steps, x_grid, u_grid)
        Vs2 = V_fds.numpy()
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD2')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :])
        plt.show()

        err = jnp.max(jnp.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_assoc_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.velocity_model()
        blocks = 50
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        block_V = hjb.directShootingValueFunction(max_iter=10)
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc 1')
        plt.show()

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.velocity_model()
        Vs2 = hjb_grid_1d_tf.seq_bw_pass_assoc(LT, blocks, x_grid, tf.constant(block_V, dtype=tf.float64))

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc 2')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :],'--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(Vs2 - Vs1))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_assoc_2(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.velocity_model()
        blocks = 50
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        block_V = hjb.directShootingValueFunction(max_iter=10)
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc 1')
        plt.show()

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.velocity_model()
        x_grid_tf = tf.constant(x_grid, dtype=tf.float64)
        print("Evaluating conditional value function (TF)...")
        block_V_tf = hjb_grid_1d_tf.direct_shooting_block_V(f, L, T / blocks, steps, x_grid_tf, max_iter=10)
        print("Done.")
        Vs2 = hjb_grid_1d_tf.seq_bw_pass_assoc(LT, blocks, x_grid_tf, block_V_tf)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc 2')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :],'--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(Vs2 - Vs1))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_assoc_par_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()
        blocks = 100
        steps = 10
        dt = T / blocks

        XT = 2.0
        S = lambda t: 1.0 - np.sqrt(2.0) * np.tanh(np.arctanh((1.0-XT) / np.sqrt(2.0)) + np.sqrt(2.0) * (t - T))
        Val_f = lambda x, t: 0.5 * S(t) * x**2

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        Vs0 = Val_f(x_mesh, t_mesh)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs0)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Exact')
        plt.show()

        J_ref = lambda s, t: 1.0 - np.sqrt(2.0) * np.tanh(np.arctanh(1.0 / np.sqrt(2.0)) + np.sqrt(2.0) * (s - t))
        A_ref = lambda s, t: 1.0 / (np.cosh(np.sqrt(2.0) * (s - t)) + (np.sqrt(2.0) * np.sinh(np.sqrt(2.0) * (s - t))) / 2.0)
        C_ref = lambda s, t: 1.0 - np.sqrt(2.0) * np.tanh(np.arctanh(1.0 / np.sqrt(2.0)) + np.sqrt(2.0) * (s - t))
        b_ref = lambda s, t: 0.0
        eta_ref = lambda s, t: 0.0

        V_ref = lambda x, s, y, t: 0.5 * J_ref(s,t) * x**2 - eta_ref(s,t) * x + 0.5 * (y - A_ref(s,t) * x - b_ref(s,t)) ** 2 / C_ref(s,t)

        xf_mesh,x0_mesh = np.meshgrid(x_grid, x_grid)
        block_V = V_ref(x0_mesh, 0.0, xf_mesh, T / blocks)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.parBackwardPass(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc par 1')
        plt.show()

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.linear_model()
        block_V_tf = tf.constant(block_V, dtype=tf.float64)
        Vs2 = hjb_grid_1d_tf.par_bw_pass(LT, blocks, x_grid, block_V_tf)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc par 2')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :], '--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(Vs2 - Vs1))
        print(err)
        self.assertTrue(err < 1e-2)

    def test_assoc_par_2(self):
        config.update("jax_enable_x64", True)
        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.velocity_model()

        blocks = 50
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.velocity_model()

        print("Evaluating conditional value function (TF)...")
        block_V_tf = hjb_grid_1d_tf.direct_shooting_block_V(f, L, T / blocks, steps, x_grid, max_iter=10)
        print("Done.")

        Vs1 = hjb_grid_1d_tf.seq_bw_pass_assoc(LT, blocks, x_grid, block_V_tf)
        Vs2 = hjb_grid_1d_tf.par_bw_pass(LT, blocks, x_grid, block_V_tf)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc seq')
        plt.show()

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc par')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :],'--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(Vs2 - Vs1))
        print(err)
        self.assertTrue(err < 1e-4)
