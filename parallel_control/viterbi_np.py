"""
Numpy-based Viterbi algorithm implementation via its associated optimal control problem
(though the parallel ones are not really implemented in parallel but simulated).

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.fsc_np as fsc_np

##############################################################################
#
# Viterbi algorithm
#
##############################################################################

class Viterbi_np:

    def __init__(self, prior, Pi, Po):
        """ Constructor.

        Parameters:
            prior: Prior state probabilities.
            Pi: State transition matrix.
            Po: Output matrix.
        """
        self.prior = prior
        self.Pi = Pi
        self.Po = Po


    def ref_viterbi(self, y_list):
        """ Reference Viterbi algorithm. This is a textbook algorithm in its plain form.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list: Viterbi value functions (probabilities).
        """
        V_list = np.zeros((y_list.shape[0]+1, self.prior.shape[0]))

        V = self.prior
        V_list[0, :] = V
        for k in range(y_list.shape[0]):
            y = y_list[k]
            psi = np.expand_dims(self.Po[:,y], 0) * self.Pi
            temp = psi * np.expand_dims(V, -1)
            V = temp.max(axis=0)
            V_list[k+1, :] = V

        v_map = np.zeros((V_list.shape[0],), dtype=y_list.dtype)
        last = True
        x = 0
        for k in reversed(range(V_list.shape[0])):
            if last:
                last = False
                x = np.argmax(V_list[k, :])
            else:
                x = np.argmax(self.Pi[:, x] * V_list[k, :])
            v_map[k] = x

        return v_map, V_list

    def ref_viterbi_log(self, y_list):
        """ Reference Viterbi algorithm. This is a textbook algorithm in its log form.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list: Viterbi value functions (log probabilities).
        """
        V_list = np.zeros((y_list.shape[0]+1, self.prior.shape[0]))

        nl_Po = -np.log(self.Po)
        nl_Pi = -np.log(self.Pi)
        V = -np.log(self.prior)
        V_list[0, :] = V
        for k in range(y_list.shape[0]):
            y = y_list[k]
            psi = np.expand_dims(nl_Po[:,y], 0) + nl_Pi
            temp = psi + np.expand_dims(V, -1)
            V = temp.min(axis=0)
            V_list[k+1, :] = V

        v_map = np.zeros((V_list.shape[0],), dtype=y_list.dtype)
        last = True
        x = 0
        for k in reversed(range(V_list.shape[0])):
            if last:
                last = False
                x = np.argmin(V_list[k, :])
            else:
                y = y_list[k]
                x = np.argmin(nl_Po[x, y] + nl_Pi[:, x] + V_list[k, :])
            v_map[k] = x

        return v_map, V_list


    def ref_viterbi_log_v2(self, y_list):
        """ Reference Viterbi algorithm. This is a textbook algorithm in its control-compatible form.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list: Viterbi value functions (log probabilities).
        """


        V_list = np.zeros((y_list.shape[0]+1, self.prior.shape[0]))
        u_list = np.zeros((y_list.shape[0], self.prior.shape[0]), dtype=int)

        nl_Po = -np.log(self.Po)
        nl_Pi = -np.log(self.Pi)
        V = -np.log(self.prior)
        u = V.argmin(axis=0)
        V_list[0, :] = V
        for k in range(y_list.shape[0]):
            y = y_list[k]
            psi = np.expand_dims(nl_Po[:,y], 0) + nl_Pi
            temp = psi + np.expand_dims(V, -1)
            V = temp.min(axis=0)
            u = temp.argmin(axis=0)
            V_list[k+1, :] = V
            u_list[k, :] = u

        v_map = np.zeros((V_list.shape[0],), dtype=y_list.dtype)
        x = np.argmin(V_list[-1, :])
        v_map[-1] = x
        for k in reversed(range(y_list.shape[0])):
            u = u_list[k, :]
            x = u[x]
            v_map[k] = x

        return v_map, V_list


    def getFSC(self, y_list):
        """ Get finite-state controller for the Viterbi problem.

        Parameters:
            y_list: Observation sequence.

        Returns:
            f: FSC object.
        """
        LT = -np.log(self.prior)
        L = []
        f = []
        n = self.prior.shape[0]
        curr_f = np.tile(np.arange(n), (n,1))
        for k in reversed(range(y_list.shape[0])):
            y = y_list[k]
            curr_L = -np.log(self.Pi.T * np.expand_dims(self.Po[:, y], -1))
            L.append(curr_L)
            f.append(curr_f)

        return fsc_np.FSC.checkAndExpand(f, L, LT)


    def fscToViterbi(self, min_x_list, V_list):
        """ Convert finite-state controller result to Viterbi solution.

        Parameters:
            min_x_list: List of states.
            V_list: List of value functions.
        """
        V_list.reverse()
        V_list_vit = np.array(V_list[:])
        V_list.reverse()

        min_x_list.reverse()
        v_map = np.array(min_x_list[:])
        min_x_list.reverse()

        return v_map, V_list_vit


    def seqViterbi(self, y_list):
        """ Sequential Viterbi algorithm using FSC.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list_vit: Viterbi value functions.
        """
        fsc = self.getFSC(y_list)

        u_list, V_list = fsc.seqBackwardPass()
        x0 = np.argmin(V_list[0])
        min_u_list, min_x_list = fsc.seqForwardPass(x0, u_list)

        v_map, V_list_vit = self.fscToViterbi(min_x_list, V_list)

        return v_map, V_list_vit


    def parBwdFwdViterbi(self, y_list):
        """ Parallel backward and forward pass Viterbi algorithm using FSC.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list_vit: Viterbi value functions.
        """
        fsc = self.getFSC(y_list)

        u_list, V_list = fsc.parBackwardPass()
        x0 = np.argmin(V_list[0])
        min_u_list, min_x_list = fsc.parForwardPass(x0, u_list)

        v_map, V_list_vit = self.fscToViterbi(min_x_list, V_list)

        return v_map, V_list_vit


    def parBwdFwdBwdViterbi(self, y_list):
        """ Parallel backward and forward-backward pass Viterbi algorithm using FSC. This is the same as
        the min-sum / max-product method.

        Parameters:
            y_list: Observation sequence.

        Returns:
            v_map: MAP state sequence.
            V_list_vit: Viterbi value functions.
        """
        fsc = self.getFSC(y_list)

        u_list, V_list = fsc.parBackwardPass()
        x0 = np.argmin(V_list[0])
        min_u_list, min_x_list = fsc.parFwdBwdPass(x0, u_list, V_list)

        v_map, V_list_vit = self.fscToViterbi(min_x_list, V_list)

        return v_map, V_list_vit
