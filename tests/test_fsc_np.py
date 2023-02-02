#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for finite state control (FSC) via dynamic programming with numpy.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.fsc_np as fsc_np

import unittest


class FSC_np_UnitTest(unittest.TestCase):
    """Unit tests for FSC_np"""

    def setupFSC(self):
        xdim = 3
        udim = 3

        rng = np.random.default_rng(123)
        track = np.array([[0,0,2,5,5,5,1,1,0],
                          [0,1,2,2,2,1,2,2,1],
                          [0,0,0,0,0,5,5,5,5]])
        track = track + 0.01 * rng.uniform(0.0,1.0,size=track.shape) # To make solution unique

        # u = 0,1,2 = left,straight,right
        f = np.array([[0,0,1],
                      [0,1,2],
                      [1,2,2]], dtype=int)
        x0 = 1

        T = track.shape[1]
        L = []
        u_cost = [1.0,0.0,1.0]
        for k in range(T):
            curr_L = np.zeros((xdim,udim))
            for x in range(xdim):
                for u in range(udim):
                    curr_L[x,u] = track[x,k] + u_cost[u]
            L.append(curr_L)

        fsc = fsc_np.FSC.checkAndExpand(f, L)

        return fsc, x0

    def test_seq(self):
        fsc, x0 = self.setupFSC()
        min_u_list1, min_x_list1, min_cost1 = fsc.batch_solution(x0)

        fsc, x0 = self.setupFSC()
        u_list, V_list = fsc.seqBackwardPass()
        min_cost2 = V_list[0][x0]

        min_u_list2, min_x_list2 = fsc.seqForwardPass(x0,u_list)

        self.assertTrue(min_x_list1 == min_x_list2)
        self.assertTrue(min_u_list1 == min_u_list2)
        self.assertTrue(abs(min_cost1 - min_cost2) < 1e-10)

    def test_par_simple(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()

        fsc, x0 = self.setupFSC()
        elems = fsc.parBackwardPass_init()

        T = len(V_list1)
        V = elems[-1]
        for k in reversed(range(T)):
            if k < T-1:
                V = fsc_np.combine_V(elems[k], V)
            self.assertTrue(linalg.norm(V[:,0] - V_list1[k]) < 1e-10)

    def test_par_backscan(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()

        fsc, x0 = self.setupFSC()
        elems = fsc.parBackwardPass_init()
        elems = fsc_np.par_backward_pass_scan(elems)

        T = len(V_list1)
        for k in range(T):
            self.assertTrue(linalg.norm(elems[k][:,0] - V_list1[k]) < 1e-10)


    def test_par_back(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()

        fsc, x0 = self.setupFSC()
        u_list2, V_list2 = fsc.parBackwardPass()

        err = max([linalg.norm(e1 - e2) for e1,e2 in zip(V_list1, V_list2)])
        self.assertTrue(err < 1e-10)
        err = max([abs(e1 - e2).max() for e1,e2 in zip(u_list1, u_list2)])
        self.assertTrue(err == 0)


    def test_par_forscan(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()
        min_u_list1, min_x_list1 = fsc.seqForwardPass(x0,u_list1)
        min_cost1 = V_list1[0][x0]

        fsc, x0 = self.setupFSC()
        u_list2, V_list2 = fsc.parBackwardPass()
        min_u_list2, min_x_list2 = fsc.parForwardPass(x0,u_list2)
        min_cost2 = V_list2[0][x0]

        self.assertTrue(min_x_list1 == min_x_list2)
        self.assertTrue(min_u_list1 == min_u_list2)
        self.assertTrue(abs(min_cost1 - min_cost2) < 1e-10)

    def test_par_fbscan(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()
        min_u_list1, min_x_list1 = fsc.seqForwardPass(x0,u_list1)
        min_cost1 = V_list1[0][x0]

        fsc, x0 = self.setupFSC()
        u_list2, V_list2 = fsc.parBackwardPass()
        min_u_list2, min_x_list2 = fsc.parFwdBwdPass(x0,u_list2,V_list2)
        min_cost2 = V_list2[0][x0]

        self.assertTrue(min_x_list1 == min_x_list2)
        self.assertTrue(min_u_list1 == min_u_list2)
        self.assertTrue(abs(min_cost1 - min_cost2) < 1e-10)

    def test_par_fbscan2(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.parBackwardPass()
        min_u_list1, min_x_list1 = fsc.parFwdBwdPass2(x0,u_list1,V_list1)
        min_cost1 = V_list1[0][x0]

        fsc, x0 = self.setupFSC()
        u_list2, V_list2 = fsc.parBackwardPass()
        min_u_list2, min_x_list2 = fsc.parFwdBwdPass(x0,u_list2,V_list2)
        min_cost2 = V_list2[0][x0]

        self.assertTrue(min_x_list1 == min_x_list2)
        self.assertTrue(min_u_list1 == min_u_list2)
        self.assertTrue(abs(min_cost1 - min_cost2) < 1e-10)


    def test_sim(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()
        min_u_list1, min_x_list1 = fsc.seqForwardPass(x0,u_list1)
        min_x_list2 = fsc.seqSimulation(x0, min_u_list1)
        self.assertTrue(min_x_list1 == min_x_list2)

    def test_cost(self):
        fsc, x0 = self.setupFSC()
        u_list1, V_list1 = fsc.seqBackwardPass()
        min_u_list1, min_x_list1 = fsc.seqForwardPass(x0,u_list1)
        min_cost1 = V_list1[0][x0]

        min_cost2 = fsc.cost(min_x_list1, min_u_list1)

#        print(min_cost1)
#        print(min_cost2)
        self.assertTrue(abs(min_cost1 - min_cost2) < 1e-10)



