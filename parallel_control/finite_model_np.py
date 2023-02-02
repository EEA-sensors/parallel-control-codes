#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-state model which represents data and discrete model for a simple game.

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.fsc_np as fsc_np


###########################################################################

class FiniteModel:
    def __init__(self, seed=123):
        self.seed = seed

    def genData(self, T, nx=21):

        rng = np.random.default_rng(self.seed)

        prob = 0.2
        nr = 2

        maski = 2
        mask  = np.array([0.5,0.3,0.1,0.3,0.5])
        track = np.ones((nx, T))
        trace = track + rng.uniform(0.0, 0.01, size=track.shape)

        x0 = nx // 2
        for r in range(nr):
            x = x0
            for k in range(T):
                u = rng.uniform(0.0,1.0)
                if u < prob:
                    if x > 0:
                        x = x - 1
                elif u > 1.0-prob:
                    if x < nx-1:
                        x = x + 1

                for i in range(mask.shape[0]):
                    y = x - maski + i
                    if y >= 0 and y < nx:
                        track[y,k] *= mask[i]

        return track, x0


    def getFSC(self, track):
        xdim = track.shape[0]
        udim = 3
        T = track.shape[1]

        f = np.zeros((xdim,udim), dtype=int)
        for x in range(xdim):
            f[x,0] = max(0,x-1)
            f[x,1] = x
            f[x,2] = min(xdim-1,x+1)
        u_cost = [1.0,0.0,1.0]

        LT = np.zeros(xdim)
        L = []
        for k in range(T):
            curr_L = np.zeros((xdim,udim))
            for x in range(xdim):
                for u in range(udim):
                    curr_L[x,u] = track[x,k] + u_cost[u]
            L.append(curr_L)

        fsc = fsc_np.FSC.checkAndExpand(f, L, LT)

        return fsc
