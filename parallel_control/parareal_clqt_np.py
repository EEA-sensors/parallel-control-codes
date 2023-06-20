"""
Parareal for CLQT backward and forward passes.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.lqt_np as lqt_np
import parallel_control.clqt_np as clqt_np
import math
import unittest

import matplotlib.pyplot as plt

##############################################################################
#
# Parareal for CLQT
#
##############################################################################

class Parareal_CLQT_np:
    def __init__(self, clqt):
        """ Constructor.

        Parameters:
            clqt: CLQT object.
        """
        self.clqt = clqt

    ###########################################################################
    # Backward pass
    ###########################################################################

    def initBackwardPass(self, blocks):
        """ Initialize backward pass.

        Parameters:
            blocks: Number of blocks.

        Returns:
            S_curr_list: List of current value function matrices.
            v_curr_list: List of current value function offsets.
        """
        dt = self.clqt.T / blocks
        _, _, S_curr_list, v_curr_list = self.clqt.seqBackwardPass(blocks, dt=dt)
        return S_curr_list, v_curr_list

    def denseBackwardPass(self, steps, S_curr_list, v_curr_list):
        """ Perform dense Parareal pass for backward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            S_curr_list: List of current value function matrices.
            v_curr_list: List of current value function offsets.

        Returns:
            S_F_list: Dense list of value function matrices.
            v_F_list: Dense list of value function offsets.
        """

        blocks = len(S_curr_list)-1

        S = S_curr_list[-1]
        v = v_curr_list[-1]

        S_F_list = [S]
        v_F_list = [v]
        dt = self.clqt.T / blocks / steps
        for k in reversed(range(blocks)):
            S = S_curr_list[k + 1]
            v = v_curr_list[k + 1]
            t1 = k * steps * dt
            _, _, S_list, v_list = self.clqt.seqBackwardPass(steps, dt=dt, t0=t1, S=S, v=v)
            S_F_list.append(S_list[0])
            v_F_list.append(v_list[0])

        S_F_list.reverse()
        v_F_list.reverse()

        return S_F_list, v_F_list

    def coarseBackwardPass(self, S_F_list, v_F_list, S_G_list, v_G_list):
        """ Perform coarse Parareal pass for backward pass of CLQT.

        Parameters:
            S_F_list: Dense list of value function matrices.
            v_F_list: Dense list of value function offsets.
            S_G_list: Coarse list of value function matrices.
            v_G_list: Coarse list of value function offsets.

        Returns:
            S_curr_list: List of current value function matrices.
            v_curr_list: List of current value function offsets.
            S_G_new_list: Coarse list of value function matrices.
            v_G_new_list: Coarse list of value function offsets.
        """

        blocks = len(S_F_list)-1

        dt = self.clqt.T / blocks

        S = S_F_list[-1]
        v = v_F_list[-1]
        S_G_new_list = [S]
        v_G_new_list = [v]
        S_curr_list = [S]
        v_curr_list = [v]
        for k in reversed(range(blocks)):
            t1 = k * dt
            _, _, S_list, v_list = self.clqt.seqBackwardPass(1, dt=dt, t0=t1, S=S, v=v)
            S_G_new = S_list[0]
            v_G_new = v_list[0]
            S = S_G_new + S_F_list[k] - S_G_list[k]
            v = v_G_new + v_F_list[k] - v_G_list[k]
            S_G_new_list.append(S_G_new)
            v_G_new_list.append(v_G_new)
            S_curr_list.append(S)
            v_curr_list.append(v)

        S_G_new_list.reverse()
        v_G_new_list.reverse()
        S_curr_list.reverse()
        v_curr_list.reverse()

        return S_curr_list, v_curr_list, S_G_new_list, v_G_new_list

    def finalBackwardPass(self, steps, S_curr_list, v_curr_list):
        """ Perform final Parareal pass for backward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            S_curr_list: List of current value function matrices.
            v_curr_list: List of current value function offsets.

        Returns:
            Kx_list: List of control gains.
            d_list: List of control offsets.
            S_list: List of value function matrices.
            v_list: List of value function offsets.
        """
        blocks = len(S_curr_list)-1

        S = S_curr_list[-1]
        v = v_curr_list[-1]

        S_list = [S]
        v_list = [v]
        Kx_list = []
        d_list = []

        dt = self.clqt.T / blocks / steps
        for k in reversed(range(blocks)):
            S = S_curr_list[k + 1]
            v = v_curr_list[k + 1]
            t1 = k * steps * dt
            bKx_list, bd_list, bS_list, bv_list = self.clqt.seqBackwardPass(steps, dt=dt, t0=t1, S=S, v=v)
            for i in range(steps):
                j = steps - i - 1
                S_list.append(bS_list[j])
                v_list.append(bv_list[j])
                Kx_list.append(bKx_list[j])
                d_list.append(bd_list[j])

        Kx_list.reverse()
        d_list.reverse()
        S_list.reverse()
        v_list.reverse()

        return Kx_list, d_list, S_list, v_list

    def backwardPass(self, blocks, steps, niter=None):
        """ Run Parareal for backward pass of CLQT.

        Parameters:
            blocks: Number of blocks.
            steps: Number of steps per block.
            niter: Number of Parareal iterations.

        Returns:
            Kx_list: List of control gains.
            d_list: List of control offsets.
            S_list: List of value function matrices.
            v_list: List of value function offsets.
        """
        if niter is None:
            niter = steps

        S_curr_list, v_curr_list = self.initBackwardPass(blocks)
        S_G_list = S_curr_list
        v_G_list = v_curr_list

        for i in range(niter):
            S_F_list, v_F_list = self.denseBackwardPass(steps, S_curr_list, v_curr_list)
            S_curr_list, v_curr_list, S_G_list, v_G_list = \
                self.coarseBackwardPass(S_F_list, v_F_list, S_G_list, v_G_list)

        return self.finalBackwardPass(steps, S_curr_list, v_curr_list)


    ###########################################################################
    # Forward pass
    ###########################################################################

    def initForwardPass(self, blocks, x0, Kx_list, d_list, u_zoh=False):
        """ Initialize Parareal for forward pass of CLQT.

        Parameters:
            blocks: Number of blocks.
            x0: Initial state.
            Kx_list: List of control gains.
            d_list: List of control offsets.
            u_zoh: Whether to use zero-order hold for control.

        Returns:
            x_curr_list: List of current states.
        """
        steps = len(Kx_list) // blocks
        _, x_curr_list = self.clqt.seqForwardPass(x0, Kx_list[::steps], d_list[::steps], u_zoh=u_zoh)
        return x_curr_list

    def denseForwardPass(self, steps, x_curr_list, Kx_list, d_list, u_zoh=False):
        """ Perform dense Parareal pass for forward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            x_curr_list: List of current states.
            Kx_list: List of control gains.
            d_list: List of control offsets.
            u_zoh: Whether to use zero-order hold for control.

        Returns:
            x_F_list: Dense list of states.
        """

        blocks = len(x_curr_list)-1

        x = x_curr_list[0]
        x_F_list = [x]

        dt = self.clqt.T / blocks / steps
        for k in range(blocks):
            x = x_curr_list[k]

            t1 = k * steps * dt
            _, x_list = self.clqt.seqForwardPass(x, Kx_list[k * steps:(k+1) * steps],
                                                 d_list[k * steps:(k+1) * steps], dt=dt,
                                                 t0=t1, u_zoh=u_zoh)

            x_F_list.append(x_list[-1])

        return x_F_list

    def coarseForwardPass(self, x_F_list, x_G_list, Kx_list, d_list, u_zoh=False):
        """ Perform coarse Parareal pass for forward pass of CLQT.

        Parameters:
            x_F_list: Dense list of states.
            x_G_list: Coarse list of states.
            Kx_list: List of control gains.
            d_list: List of control offsets.
            u_zoh: Whether to use zero-order hold for control.

        Returns:
            x_curr_list: List of current states.
            x_G_new_list: Coarse list of states.
        """

        blocks = len(x_F_list)-1
        steps = len(Kx_list) // blocks

        dt = self.clqt.T / blocks

        x = x_F_list[0]
        x_G_new_list = [x]
        x_curr_list = [x]

        for k in range(blocks):
            t1 = k * dt

            _, x_list = self.clqt.seqForwardPass(x, Kx_list[k * steps:k * steps + 1],
                                                 d_list[k * steps:k * steps + 1], dt=dt,
                                                 t0=t1, u_zoh=u_zoh)

            x_G_new = x_list[-1]
            x = x_G_new + x_F_list[k + 1] - x_G_list[k + 1]
            x_G_new_list.append(x_G_new)
            x_curr_list.append(x)

        return x_curr_list, x_G_new_list

    def finalForwardPass(self, steps, x_curr_list, Kx_list, d_list, u_zoh=False):
        """ Perform final Parareal pass for forward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            x_curr_list: List of current states.
            Kx_list: List of control gains.
            d_list: List of control offsets.
            u_zoh: Whether to use zero-order hold for control.

        Returns:
            u_list: List of control inputs.
            x_list: List of states.
        """

        blocks = len(x_curr_list)-1

        x_list = []
        u_list = []

        dt = self.clqt.T / blocks / steps
        for k in range(blocks):
            x = x_curr_list[k]

            t1 = k * steps * dt
            bu_list, bx_list = self.clqt.seqForwardPass(x, Kx_list[k * steps:(k+1) * steps],
                                                        d_list[k * steps:(k+1) * steps], dt=dt,
                                                        t0=t1, u_zoh=u_zoh)

            for i in range(steps):
                x_list.append(bx_list[i])
                u_list.append(bu_list[i])

        x_list.append(bx_list[-1])

        return u_list, x_list

    def forwardPass(self, blocks, steps, x0, Kx_list, d_list, u_zoh=False, niter=None):
        """ Perform Parareal pass for forward pass of CLQT.

        Parameters:
            blocks: Number of blocks.
            steps: Number of steps per block.
            x0: Initial state.
            Kx_list: List of control gains.
            d_list: List of control offsets.
            u_zoh: Whether to use zero-order hold for control.
            niter: Number of iterations.

        Returns:
            u_list: List of control inputs.
            x_list: List of states.
        """
        if niter is None:
            niter = steps

        x_curr_list = self.initForwardPass(blocks, x0, Kx_list, d_list, u_zoh=u_zoh)
        x_G_list = x_curr_list

        for i in range(niter):
            x_F_list = self.denseForwardPass(steps, x_curr_list, Kx_list, d_list, u_zoh=u_zoh)
            x_curr_list, x_G_list = \
                self.coarseForwardPass(x_F_list, x_G_list, Kx_list, d_list, u_zoh=u_zoh)

        return self.finalForwardPass(steps, x_curr_list, Kx_list, d_list, u_zoh=u_zoh)

    ###########################################################################
    # Forward-backward pass
    ###########################################################################

    def initFwdBwdPass(self, blocks, x0):
        """ Perform initial Parareal pass for forward-backward pass of CLQT.

        Parameters:
            blocks: Number of blocks.
            x0: Initial state.

        Returns:
            A_curr_list: List of current A matrices.
            b_curr_list: List of current b vectors.
            C_curr_list: List of current C matrices.
        """

        dt = self.clqt.T / blocks
        A_curr_list, b_curr_list, C_curr_list = self.clqt.seqFwdBwdPass(x0, blocks, dt=dt)
        return A_curr_list, b_curr_list, C_curr_list

    def denseFwdBwdPass(self, steps, A_curr_list, b_curr_list, C_curr_list):
        """ Perform dense Parareal pass for forward-backward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            A_curr_list: List of current A matrices.
            b_curr_list: List of current b vectors.
            C_curr_list: List of current C matrices.

        Returns:
            A_F_list: Dense list of A matrices.
            b_F_list: Dense list of b vectors.
            C_F_list: Dense list of C matrices.
        """
        blocks = len(A_curr_list)-1

        A = A_curr_list[0]
        b = b_curr_list[0]
        C = C_curr_list[0]
        A_F_list = [A]
        b_F_list = [b]
        C_F_list = [C]
        dt = self.clqt.T / blocks / steps
        for k in range(blocks):
            A = A_curr_list[k]
            b = b_curr_list[k]
            C = C_curr_list[k]

            t1 = k * steps * dt
            A_list, b_list, C_list = self.clqt.seqFwdBwdPass(None, steps, dt=dt, t0=t1, A=A, b=b, C=C)

            A_F_list.append(A_list[-1])
            b_F_list.append(b_list[-1])
            C_F_list.append(C_list[-1])

        return A_F_list, b_F_list, C_F_list

    def coarseFwdBwdPass(self, A_F_list, b_F_list, C_F_list, A_G_list, b_G_list, C_G_list):
        """ Perform coarse Parareal pass for forward-backward pass of CLQT.

        Parameters:
            A_F_list: Dense list of A matrices.
            b_F_list: Dense list of b vectors.
            C_F_list: Dense list of C matrices.
            A_G_list: Coarse list of A matrices.
            b_G_list: Coarse list of b vectors.
            C_G_list: Coarse list of C matrices.

        Returns:
            A_curr_list: List of current A matrices.
            b_curr_list: List of current b vectors.
            C_curr_list: List of current C matrices.
            A_G_new_list: New coarse list of A matrices.
            b_G_new_list: New coarse list of b vectors.
            C_G_new_list: New coarse list of C matrices.
        """
        blocks = len(A_F_list)-1

        A = A_F_list[0]
        b = b_F_list[0]
        C = C_F_list[0]
        A_G_new_list = [A]
        b_G_new_list = [b]
        C_G_new_list = [C]
        A_curr_list = [A]
        b_curr_list = [b]
        C_curr_list = [C]
        dt = self.clqt.T / blocks
        for k in range(blocks):
            t1 = k * dt

            A_list, b_list, C_list = self.clqt.seqFwdBwdPass(None, 1, dt=dt, t0=t1, A=A, b=b, C=C)

            A_G_new = A_list[-1]
            b_G_new = b_list[-1]
            C_G_new = C_list[-1]
            A = A_G_new + A_F_list[k + 1] - A_G_list[k + 1]
            b = b_G_new + b_F_list[k + 1] - b_G_list[k + 1]
            C = C_G_new + C_F_list[k + 1] - C_G_list[k + 1]
            A_G_new_list.append(A_G_new)
            b_G_new_list.append(b_G_new)
            C_G_new_list.append(C_G_new)
            A_curr_list.append(A)
            b_curr_list.append(b)
            C_curr_list.append(C)

        return A_curr_list, b_curr_list, C_curr_list, A_G_new_list, b_G_new_list, C_G_new_list

    def finalFwdBwdPass(self, steps, A_curr_list, b_curr_list, C_curr_list):
        """ Perform final Parareal pass for forward-backward pass of CLQT.

        Parameters:
            steps: Number of steps per block.
            A_curr_list: List of current A matrices.
            b_curr_list: List of current b vectors.
            C_curr_list: List of current C matrices.

        Returns:
            A_list: Dense list of A matrices.
            b_list: Dense list of b vectors.
            C_list: Dense list of C matrices.
        """
        blocks = len(A_curr_list)-1

        A = A_curr_list[0]
        b = b_curr_list[0]
        C = C_curr_list[0]
        A_list = []
        b_list = []
        C_list = []
        dt = self.clqt.T / blocks / steps
        for k in range(blocks):
            A = A_curr_list[k]
            b = b_curr_list[k]
            C = C_curr_list[k]

            t1 = k * steps * dt
            bA_list, bb_list, bC_list = self.clqt.seqFwdBwdPass(None, steps, dt=dt, t0=t1, A=A, b=b, C=C)

            for i in range(steps):
                A_list.append(bA_list[i])
                b_list.append(bb_list[i])
                C_list.append(bC_list[i])

        A_list.append(bA_list[-1])
        b_list.append(bb_list[-1])
        C_list.append(bC_list[-1])

        return A_list, b_list, C_list

    def fwdBwdPass(self, blocks, steps, x0, niter=None):
        """ Perform forward-backward pass of CLQT.

        Parameters:
            blocks: Number of blocks.
            steps: Number of steps per block.
            x0: Initial state.
            niter: Number of Parareal iterations.

        Returns:
            A_list: Dense list of A matrices.
            b_list: Dense list of b vectors.
            C_list: Dense list of C matrices.
        """
        if niter is None:
            niter = steps

        A_curr_list, b_curr_list, C_curr_list = self.initFwdBwdPass(blocks, x0)
        A_G_list = A_curr_list
        b_G_list = b_curr_list
        C_G_list = C_curr_list

        for i in range(niter):
            A_F_list, b_F_list, C_F_list = self.denseFwdBwdPass(steps, A_curr_list, b_curr_list, C_curr_list)
            A_curr_list, b_curr_list, C_curr_list, A_G_list, b_G_list, C_G_list = \
                self.coarseFwdBwdPass(A_F_list, b_F_list, C_F_list, A_G_list, b_G_list, C_G_list)

        return self.finalFwdBwdPass(steps, A_curr_list, b_curr_list, C_curr_list)


