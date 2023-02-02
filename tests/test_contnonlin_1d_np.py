#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Numpy/Jax versions of routines for solving continuous-time nonlinear 1d problems.

@author: Simo Särkkä
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize
import parallel_control.hjb_grid_1d_np as cnl1d_np

class Cnl1d_np_UnitTest(unittest.TestCase):
    def test_bfgs(self):

        def f(x):
            return jnp.sum((x-1.0)**2)

        options = dict(disp=True, maxiter=10, gtol=1e-6)
        jacobian = jax.jacfwd(f)
#        x = scipy.optimize.minimize(f, jnp.array([0.1]), method="BFGS", options=options)
        x = scipy.optimize.minimize(f, 0.1, method="BFGS", options=options)
        print(x)



