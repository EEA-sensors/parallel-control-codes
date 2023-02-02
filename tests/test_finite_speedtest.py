#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for finite speedtests.

@author: Simo Särkkä
"""

import unittest
import tensorflow as tf
import parallel_control.finite_speedtest as fscspeed

class FiniteSpeedtest_UnitTest(unittest.TestCase):
    def test_fsc_seq_bw_speedtest(self):
        fsc_gen = fscspeed.finite_generator(31, 2, 3, 3)
        _ = fscspeed.fsc_seq_bw_speedtest(fsc_gen)

    def test_fsc_par_bw_speedtest(self):
        fsc_gen = fscspeed.finite_generator(31, 2, 3, 3)
        _ = fscspeed.fsc_par_bw_speedtest(fsc_gen)

    def test_fsc_seq_bwfw_speedtest(self):
        fsc_gen = fscspeed.finite_generator(31, 2, 3, 3)
        _ = fscspeed.fsc_seq_bwfw_speedtest(fsc_gen)

    def test_fsc_par_bwfw_1_speedtest(self):
        fsc_gen = fscspeed.finite_generator(31, 2, 3, 3)
        _ = fscspeed.fsc_par_bwfw_1_speedtest(fsc_gen)

    def test_fsc_par_bwfw_2_speedtest(self):
        fsc_gen = fscspeed.finite_generator(31, 2, 3, 3)
        _ = fscspeed.fsc_par_bwfw_2_speedtest(fsc_gen)
