"""
Gilbert-Elliot model for Viterbi testing.

@author: Simo Särkkä
"""

import numpy as np

###########################################################################
#
# Utility functions
#
###########################################################################

def get_b_list(x_list):
    """ Get b list from x list.

    Parameters:
        x_list: State list.

    Returns:
        b_list: Variable b list.
    """
    b_list = np.zeros(x_list.shape, dtype=x_list.dtype)
    for i in range(x_list.shape[0]):
        if x_list[i] == 2 or x_list[i] == 3:
            b_list[i] = 1

    return b_list

###########################################################################
#
# Gilbert-Elliot model for Viterbi testing.
#
###########################################################################


class GEModel:
    def __init__(self, seed=123):
        """ Constructor for Gilbert-Elliot model.

        Parameters:
            seed: Random seed.
        """
        self.seed = seed
        p0 = 0.03  # P(S_{k + 1} = 1 | S_{k} = 0)
        p1 = 0.1   # P(S_{k + 1} = 0 | S_{k} = 1)
        p2 = 0.05  # P(B_{k + 1} = 1 | B_{k} = 0) = P(B_{k + 1} = 0 | B_{k} = 1)
        q0 = 0.1   # P(Y_k != b | B_k = b, S_k = 0)
        q1 = 0.3   # P(Y_k != b | B_k = b, S_k = 1)

        Pi = np.array([[(1.0 - p0) * (1.0 - p2), p0 * (1.0 - p2), (1.0 - p0) * p2, p0 * p2],
                       [p1 * (1.0 - p2), (1.0 - p1) * (1.0 - p2), p1 * p2, (1.0 - p1) * p2],
                       [(1.0 - p0) * p2, p0 * p2, (1.0 - p0) * (1.0 - p2), p0 * (1.0 - p2)],
                       [p1 * p2, (1.0 - p1) * p2, p1 * (1.0 - p2), (1.0 - p1) * (1.0 - p2)]])

        Po = np.array([[(1.0 - q0), q0],
                      [(1.0 - q1), q1],
                      [q0, (1.0 - q0)],
                      [q1, (1.0 - q1)]])

        prior = np.array([1.0, 1.0, 1.0, 1.0]) / 4.0
#        prior = np.array([0.8, 0.9, 1.1, 1.2]) / 4.0

        self.Pi = Pi
        self.Po = Po
        self.prior = prior

    def genData(self, T):
        """ Generate data from the model.

        Parameters:
            T: Length of data.

        Returns:
            x_list: State list.
            y_list: Observation list.
        """
        np.random.seed(self.seed)

        def categ_rnd(probs, N=1):
            return np.random.choice(a=probs.shape[0], size=N, p=probs)

        x_list = np.zeros((T,), dtype=int)
        y_list = np.zeros((T,), dtype=int)

        x = categ_rnd(self.prior)[0]
        for k in range(T):
            x = categ_rnd(self.Pi[x,:])[0]
            y = categ_rnd(self.Po[x,:])[0]

            x_list[k] = x
            y_list[k] = y

        return x_list, y_list

