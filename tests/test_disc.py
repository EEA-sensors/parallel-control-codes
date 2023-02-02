#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for discretization routines.

@author: Simo Särkkä
"""

import unittest
import numpy as np
from scipy import linalg
import parallel_control.disc as disc

class Disc_UnitTest(unittest.TestCase):
    def test_lti_disc(self):
        F  = np.array([[0.0,1.0],[0.0,0.0]])
        Qc = 0.1
        L  = np.array([[0],[1]])
        dt = 2.0

        A, Q = disc.lti_disc(F, L, Qc, dt)

        A1 = np.array([[1.0, dt], [0.0, 1.0]])
        Q1 = Qc * np.array([[dt**3 / 3.0, dt**2  / 2.0], [dt**2 / 2.0, dt]])

        self.assertTrue(linalg.norm((A-A1)) < 1e-10)
        self.assertTrue(linalg.norm((Q-Q1)) < 1e-10)


    def test_lti_disc_u(self):
        F = np.array([[0.0, 1.0], [0.0, 0.0]])
        Qc = 0.1
        L = np.array([[0], [1]])
        G = np.array([[0], [2]])
        dt = 0.5

        A, B, Q = disc.lti_disc_u(F, L, G, Qc, dt)

        A1 = np.array([[1.0, dt], [0.0, 1.0]])
        B1 = np.array([[dt**2],[2.0*dt]])
        Q1 = Qc * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])

        self.assertTrue(linalg.norm((A - A1)) < 1e-10)
        self.assertTrue(linalg.norm((B - B1)) < 1e-10)
        self.assertTrue(linalg.norm((Q - Q1)) < 1e-10)

