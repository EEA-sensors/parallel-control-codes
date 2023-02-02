#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for nonlinear speedtests.

@author: Simo Särkkä
"""

import unittest
import tensorflow as tf
import parallel_control.nonlinear_speedtest as nlinspeed

class NonlinearSpeedtest_UnitTest(unittest.TestCase):
    def test_nlqt_iter_seq_speedtest(self):
        nlqt_gen = nlinspeed.nonlinear_generator(2, 3, 3)
        _ = nlinspeed.nlqt_iter_seq_speedtest(nlqt_gen)

    def test_iter_par_1_seq_speedtest(self):
        nlqt_gen = nlinspeed.nonlinear_generator(2, 3, 3)
        _ = nlinspeed.nlqt_iter_par_1_speedtest(nlqt_gen)

    def test_iter_par_2_seq_speedtest(self):
        nlqt_gen = nlinspeed.nonlinear_generator(2, 3, 3)
        _ = nlinspeed.nlqt_iter_par_2_speedtest(nlqt_gen)
