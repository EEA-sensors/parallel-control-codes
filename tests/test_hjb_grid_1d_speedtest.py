"""
Unit tests for HJB 1d speedtest routines.

@author: Simo Särkkä
"""

import unittest
import tensorflow as tf
import parallel_control.hjb_grid_1d_speedtest as hjbspeed
import math

class HJBSpeedtest_UnitTest(unittest.TestCase):
    def test_upwind_1(self):
        model = hjbspeed.get_model(20, 20)
        blocks = 100
        for i in [1,2,3]:
            elapsed, err1, err2 = hjbspeed.seq_upwind_speedtest(model, i * blocks)

    def test_seq_assoc_1(self):
        model = hjbspeed.get_model(20, 20)
        blocks = 100
        for i in [1,2,3]:
            elapsed, err1, err2 = hjbspeed.seq_assoc_speedtest(model, i * blocks)

    def test_par_assoc_1(self):
        model = hjbspeed.get_model(20, 20)
        blocks = 100
        for i in [1,2,3]:
            elapsed, err1, err2 = hjbspeed.par_assoc_speedtest(model, i * blocks)

