#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Numpy/Jax versions of routines for solving continuous-time nonlinear 1d problems.

@author: Simo Särkkä
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.config import config

import parallel_control.hjb_grid_1d_np as hjb_grid_1d_np
import parallel_control.cnonlin_models_np as cnonlin_models_np
import parallel_control.fsc_np as fsc_np
import parallel_control.clqt_np as clqt_np

import math
import unittest


##############################################################################
# Unit tests for HJB on grid
##############################################################################

class HJB_Grid_1d_np_UnitTest(unittest.TestCase):
    def test_combine_1(self):
        N = 5
        Va = np.random.normal(0.0, 1.0, (N,N))
        Vb = np.random.normal(0.0, 1.0, (N,N))
        V0 = fsc_np.combine_V(Va, Vb)
        V1 = hjb_grid_1d_np.combine_V2(Va, Vb)
        V2 = hjb_grid_1d_np.combine_V(Va, Vb)

#        print(V1)
#        print(V2)
        self.assertTrue(np.linalg.norm(V1 - V2) < 1e-5)
        self.assertTrue(np.linalg.norm(V0 - V2) < 1e-5)

    def test_combine_2(self):
        N = 5
        Va = np.random.normal(0.0, 1.0, (N, N))
        Vb = np.random.normal(0.0, 1.0, (N, N))
        V1 = hjb_grid_1d_np.combine_V_interp2(Va, Vb)
        V2 = hjb_grid_1d_np.combine_V_interp(Va, Vb)

        print(V1)
        print(V2)
        self.assertTrue(np.linalg.norm(V1 - V2) < 1e-5)

    def test_direct_shooting_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.linear_model()

        J_ref = lambda s, t: 1.0 - np.sqrt(2.0) * np.tanh(np.arctanh(1.0 / np.sqrt(2.0)) + np.sqrt(2.0) * (s - t))
        A_ref = lambda s, t: 1.0 / (np.cosh(np.sqrt(2.0) * (s - t)) + (np.sqrt(2.0) * np.sinh(np.sqrt(2.0) * (s - t))) / 2.0)
        C_ref = lambda s, t: 1.0 - np.sqrt(2.0) * np.tanh(np.arctanh(1.0 / np.sqrt(2.0)) + np.sqrt(2.0) * (s - t))
        b_ref = lambda s, t: 0.0
        eta_ref = lambda s, t: 0.0

        V_ref = lambda x, s, y, t: 0.5 * J_ref(s,t) * x**2 - eta_ref(s,t) * x + 0.5 * (y - A_ref(s,t) * x - b_ref(s,t)) ** 2 / C_ref(s,t)

        T = 1.0

        blocks = 5
        x_grid = np.linspace(-2.0, 2.0, 20)
        xf_mesh,x0_mesh = np.meshgrid(x_grid, x_grid)
        z_mesh = V_ref(x0_mesh, 0.0, xf_mesh, T / blocks)

        fig, ax = plt.subplots()
        p = ax.pcolor(x0_mesh, xf_mesh, z_mesh)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Exact')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, 10, x_grid)
        V = hjb.directShootingValueFunction()

        fig, ax = plt.subplots()
        p = ax.pcolor(x0_mesh, xf_mesh, V)
        cb = fig.colorbar(p, ax=ax)
        plt.title('DS')
        plt.show()

        fig, ax = plt.subplots()
        p = ax.pcolor(x0_mesh, xf_mesh, V-z_mesh)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Difference')
        plt.show()

        err = jnp.max(jnp.abs(V - z_mesh))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_symfd_1(self):
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

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_symfd(u_grid)
        Vs1 = jnp.array(V_list)

        # This is inaccurate, but no can do
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD')
        plt.show()

        plt.plot(x_grid, Vs0[0, :])
        plt.plot(x_grid, Vs1[0, :])
        plt.show()

        # Check only the middle part which is accurate
        ind1 = x_grid.shape[0] // 4
        ind2 = 3 * x_grid.shape[0] // 4

        err = jnp.max(jnp.abs(Vs0[:,ind1:ind2] - Vs1[:,ind1:ind2]))
        print(err)
        self.assertTrue(err < 1e-2)


    def test_upwind_1(self):
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

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_upwind(u_grid)
        Vs1 = jnp.array(V_list)

        # This is inaccurate, but no can do
        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Upwind')
        plt.show()

        plt.plot(x_grid, Vs0[0, :])
        plt.plot(x_grid, Vs1[0, :])
        plt.show()

        # Check only the middle part which is accurate
        ind1 = x_grid.shape[0] // 4
        ind2 = 3 * x_grid.shape[0] // 4

        err = jnp.max(jnp.abs(Vs0[:,ind1:ind2] - Vs1[:,ind1:ind2]))
        print(err)
        self.assertTrue(err < 1e-1)

    def test_upwind_2(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.upwind_model()
        blocks = 100
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        V_ref = lambda x, t: (x > 0.0) * (-x * jnp.exp(T-t)) + (x <= 0.0) * (-x)
        Vs0 = V_ref(x_mesh, t_mesh)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs0)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Exact')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_upwind(u_grid)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Upwind')
        plt.show()


    def test_assoc_1(self):
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
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc')
        plt.show()

        plt.plot(x_grid, Vs0[0, :])
        plt.plot(x_grid, Vs1[0, :],'--')
        plt.show()

    def test_assoc_2(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.upwind_model(100, 100)
        blocks = 10
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        V_ref = lambda x, t: (x > 0.0) * (-x * jnp.exp(T-t)) + (x <= 0.0) * (-x)
        Vs0 = V_ref(x_mesh, t_mesh)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs0)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Exact')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        block_V = cnonlin_models_np.upwind_model_block_V(hjb_grid_1d_np._HJB_GRID_NP_INFTY, dt, x_grid)
        print(block_V)
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc')
        plt.show()

    def test_upwind_assoc_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.velocity_model(20, 20)
        T = 1.0
        blocks = 10
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T+dt, blocks+1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_upwind(u_grid)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Upwind')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        block_V = hjb.directShootingValueFunction(max_iter=10)
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs2 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc')
        plt.show()

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2-Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Difference')
        plt.show()

        fig, ax = plt.subplots()
        xf_mesh, x0_mesh = jnp.meshgrid(x_grid, x_grid)
        p = ax.pcolor(x0_mesh, xf_mesh, block_V)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Block V')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :],'--')
        plt.show()


        err1 = jnp.max(jnp.abs(Vs2 - Vs1))
        print(err1)

        n = x_grid.shape[0]
        err2 = jnp.max(jnp.abs(Vs2[:, (n // 3):(2*n // 3)] - Vs1[:, (n // 3):(2*n // 3)]))
        print(err2)

    def test_symfd_assoc_1(self):
        config.update("jax_enable_x64", True)

        f, L, LT, T, x_grid, u_grid = cnonlin_models_np.velocity_model()
        blocks = 50
        steps = 10
        dt = T / blocks

        t_grid = jnp.linspace(0, T + dt, blocks + 1, dtype=jnp.float64)
        x_mesh, t_mesh = jnp.meshgrid(x_grid, t_grid)

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.seqBackwardPass_symfd(u_grid)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('SymFD')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, 10, x_grid)
        block_V = hjb.directShootingValueFunction()
        print(block_V)
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs2 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :],'--')
        plt.show()

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
        V_list = hjb.seqBackwardPass_assoc(block_V)
        Vs1 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs1)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc seq')
        plt.show()

        hjb = hjb_grid_1d_np.HJB_Grid_1d(f, L, LT, T, blocks, steps, x_grid)
        V_list = hjb.parBackwardPass(block_V)
        Vs2 = jnp.array(V_list)

        fig, ax = plt.subplots()
        p = ax.pcolor(t_mesh, x_mesh, Vs2)
        cb = fig.colorbar(p, ax=ax)
        plt.title('Assoc par')
        plt.show()

        plt.plot(x_grid, Vs1[0, :])
        plt.plot(x_grid, Vs2[0, :], '--')
        plt.show()
