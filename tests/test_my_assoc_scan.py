#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for manual associative scan implementation.

@author: Simo Särkkä
"""

from parallel_control.my_assoc_scan import my_assoc_scan
import unittest

class Scan_UnitTest(unittest.TestCase):

    def test_fwd(self):
        a = [4, 1, 2, 3, 5, 3]
        a = my_assoc_scan(lambda x, y: x + y, a)
        self.assertEqual(a, [4, 5, 7, 10, 15, 18])

    def test_bwd(self):
        a = [4, 1, 2, 3, 5, 3]
        a = my_assoc_scan(lambda x, y: x + y, a, reverse=True)
        self.assertEqual(a, [18, 14, 13, 11, 8, 3])


