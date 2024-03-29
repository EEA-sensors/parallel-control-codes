"""
Numpy-based optimal finite state control (FSC) via dynamic programming.
Both sequential and (simulated) parallel versions.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
from parallel_control.my_assoc_scan import my_assoc_scan

##############################################################################
#
# Misc. constants and utilities
#
##############################################################################

_FSC_NP_INFTY = 1e20 # This is used as "infinity" in computations

##############################################################################
#
# Combination functions and general associative scan for FSC
#
##############################################################################

def combine_V(Vij, Vjk):
    """ Combine two value functions Vij and Vjk into Vik.

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
    Vik = np.zeros_like(Vij)
    for i in range(Vij.shape[0]):
        for k in range(Vjk.shape[1]):
            Vik[i,k] = (Vij[i,:] + Vjk[:,k]).min()

    return Vik

def combine_f(fij, fjk):
    """ Combine two forward functions fij and fjk into fik.

    Parameters:
        fij: Forward function for i -> j
        fjk: Forward function for j -> k

    Returns:
        fik: Forward function for i -> k
    """
#    fik = np.zeros_like(fij)
#    for i in range(fij.shape[0]):
#        fik[i] = fjk[fij[i]]
    fik = fjk[fij]

    return fik


def par_backward_pass_scan(elems):
    """ Run associative scan for parallel backward pass.

    Parameters:
        elems: List of conditional value function elements

    Returns:
        List of prefix-summed conditional value function elements
    """
    return my_assoc_scan(lambda x, y: combine_V(x, y), elems, reverse=True)

def par_forward_pass_scan(elems):
    """ Run associative scan for parallel backward pass.

    Parameters:
        elems: List of forward function elements

    Returns:
        List of prefix-summed forward function elements
    """
    return my_assoc_scan(lambda x, y: combine_f(x, y), elems, reverse=False)

def par_fwdbwd_pass_scan(elems):
    """ Run associative scan for parallel forward pass for value functions.

    Parameters:
        elems: List of conditional value function elements

    Returns:
        List of prefix-summed conditional value function elements
    """
    return my_assoc_scan(lambda x, y: combine_V(x, y), elems, reverse=False)


##############################################################################
#
# Sequential and parallel FSC
#
##############################################################################

class FSC:
    
    def __init__(self, f, L, LT):
        """ Constructor.

        Parameters:
            f: Forward functions as list of matrices f[x,u].
            L: Conditional value functions as list of matrices L[x,u].
            LT: Terminal value function as matrix LT[x].
        """
        self.f  = f
        self.L  = L
        self.LT = LT
        self.gamma = 1.0 # TODO: Not yet supported in the parallel version

    @classmethod
    def checkAndExpand(cls, f, L, LT=None, T=None):
        """
        Create FSC object from given parameters.
        Check that all dimension match and then convert all the indexed
        parameters into lists of length T by replication if they are not
        already.

        Parameters:
            f: Forward functions matrix or list of matrices f[x,u].
            L: Conditional value functions matrix or list of matrices L[x,u].
            LT: Terminal value function matrix LT[x] (default ones).
            T: Number of time steps (default None in which case deduced).

        Returns:
            FSC object.
        """
        if isinstance(f, list):
            T = len(f)
        elif isinstance(L, list):
            T = len(L)
        else:
            if T is None:
                raise ValueError("Parameter T cannot None when f and L are matrices.")

        if not isinstance(f, list):
            f = T * [f]
        if not isinstance(L, list):
            L = T * [L]
        if len(f) != len(L):
            raise ValueError(f"Lengths of f ({len(f)}) and L ({len(L)}) lists don't match.")

        xdim = f[0].shape[0]
        udim = f[0].shape[1]

        if LT is None:
            LT = np.ones((xdim,), dtype=L[0].dtype)

        for k in range(T):
            if (f[k].shape[0] != xdim) or (f[k].shape[1] != udim):
                raise ValueError(f"Shape of f[{k}] ({f[k].shape}) is not as expected ({xdim},{udim}).")
            if (L[k].shape[0] != xdim) or (L[k].shape[1] != udim):
                raise ValueError(f"Shape of L[{k}] ({L[k].shape}) is not as expected ({xdim},{udim}).")

        if f[0].shape[0] != LT.shape[0]:
            raise ValueError(f"Shapes of f[0] ({f[0].shape}) and LT ({LT.shape}) don't match.")

        return cls(f, L, LT)

    ###########################################################################
    # Sequential FSC routines
    ###########################################################################

    def seqBackwardPass(self):
        """ Run sequential backward pass.

        Returns:
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.
        """
        T = len(self.f)
        V = self.LT
        
        V_list  = [V]
        u_list = []
        
        for k in reversed(range(T)):
            L  = self.L[k]
            f  = self.f[k]
            Vu = np.zeros_like(L)
            for u in range(Vu.shape[1]):
                Vu[:,u] = L[:,u] + self.gamma * V[f[:,u]]
            u = Vu.argmin(axis=1)
            V = Vu.min(axis=1)
            
            u_list.append(u)
            V_list.append(V)

        u_list.reverse()
        V_list.reverse()
        
        return u_list, V_list
    

    def seqForwardPass(self, x0, u_list):
        """ Run sequential forward pass.

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        T = len(u_list)

        x = x0
        min_u_list = []
        min_x_list = [x]
        
        for k in range(T):
            u = u_list[k][x]
            f = self.f[k]
            x = f[x,u]
            min_u_list.append(u)
            min_x_list.append(x)
            
        return min_u_list, min_x_list


    def seqSimulation(self, x0, u_list):
        """ Run sequential simulation.

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.

        Returns:
            x_list: List of states for k=0,...,T.
        """
        T = len(u_list)

        x = x0
        x_list = [x]

        for k in range(T):
            u = u_list[k]
            f = self.f[k]
            x = f[x, u]
            x_list.append(x)

        return x_list

    ###########################################################################
    # Parallel backward pass
    ###########################################################################

    def parBackwardPass_init(self):
        """ Initialize parallel backward pass elements.

        Returns:
            elems: List of elements to be used in parallel backward pass.
        """
        T = len(self.f)
        xdim = self.f[0].shape[0]
        udim = self.f[0].shape[1]
        elems = []

        for k in range(T):
            V = np.full((xdim,xdim), _FSC_NP_INFTY)
            L = self.L[k]
            f = self.f[k]
            for x in range(xdim):
                for u in range(udim):
                    v  = L[x,u]
                    xp = f[x,u]
                    if v < V[x,xp]:
                        V[x,xp] = v
            elems.append(V)

        V = np.full((xdim,xdim), _FSC_NP_INFTY)
        for x in range(xdim):
            for xp in range(xdim):
                V[x,xp] = self.LT[x]
        elems.append(V)

        return elems

    def parBackwardPass_extract(self, elems):
        """ Extract results from parallel backward pass elements.

        Parameters:
            elems: List of elements from the parallel backward pass.

        Returns:
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.
        """

        V_list = [elems[0][:,0]]
        u_list = []
        for k in range(len(elems)-1):
            V = elems[k+1][:,0]
            V_list.append(V)
            L  = self.L[k]
            f  = self.f[k]
            Vu = np.zeros_like(L)
            for u in range(Vu.shape[1]):
                Vu[:,u] = L[:,u] + V[f[:,u]]
            u = Vu.argmin(axis=1)
            u_list.append(u)

        return u_list, V_list

    def parBackwardPass(self):
        """ Run parallel backward pass.

        Returns:
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.
        """
        elems = self.parBackwardPass_init()
        elems = par_backward_pass_scan(elems)
        u_list, V_list = self.parBackwardPass_extract(elems)
        return u_list, V_list

    ###########################################################################
    # Parallel forward pass with function decomposition
    ###########################################################################

    def parForwardPass_init(self, x0, u_list):
        """ Initialize parallel forward pass elements.

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.

        Returns:
            elems: List of elements to be used in parallel forward pass.
        """
        T = len(self.f)
        xdim = self.f[0].shape[0]
        elems = []

        e = np.zeros_like(self.f[0][:,0])
        for i in range(len(e)):
            e[i] = x0
        elems.append(e)

        for k in range(T):
            u = u_list[k]
            f = self.f[k]
            e = np.zeros_like(f[:,0])
            for x in range(xdim):
                e[x] = f[x,u[x]]
            elems.append(e)

        return elems

    def parForwardPass_extract(self, elems, u_list):
        """ Extract results from parallel forward pass elements.

        Parameters:
            elems: List of elements from the parallel forward pass.
            u_list: List of optimal actions vectors for k=0,...,T-1.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        min_u_list = []
        min_x_list = []

        for k in range(len(elems)):
            x = elems[k][0]
            min_x_list.append(x)
            if k < len(u_list):
                u = u_list[k][x]
                min_u_list.append(u)

        return min_u_list, min_x_list

    def parForwardPass(self, x0, u_list):
        """ Run parallel forward pass.

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        elems = self.parForwardPass_init(x0,u_list)
        elems = par_forward_pass_scan(elems)
        min_u_list, min_x_list = self.parForwardPass_extract(elems,u_list)

        return min_u_list, min_x_list

    ###########################################################################
    # Parallel forward pass with value function composition with auxiliary
    # intialization for x0
    ###########################################################################

    def parFwdBwdPass_init(self,x0):
        """ Initialize parallel forward-backward pass forward elements.

        Parameters:
            x0: Initial state.

        Returns:
            elems: List of elements to be used in parallel value function forward pass.
        """
        T = len(self.f)
        xdim = self.f[0].shape[0]
        udim = self.f[0].shape[1]

        elems = []

        V = np.full((xdim, xdim), _FSC_NP_INFTY)
        V[:,x0] = 0
        elems.append(V)

        for k in range(T):
            V = np.full((xdim,xdim), _FSC_NP_INFTY)
            L = self.L[k]
            f = self.f[k]
            for x in range(xdim):
                for u in range(udim):
                    v  = L[x,u]
                    xp = f[x,u]
                    if v < V[x,xp]:
                        V[x,xp] = v
            elems.append(V)

        return elems

    def parFwdBwdPass_extract(self,elems,u_list,V_list):
        """ Extract results from parallel forward-backward pass elements.

        Parameters:
            elems: List of elements from the parallel forward-backward pass.
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        min_u_list = []
        min_x_list = []

        for k in range(len(elems)):
            Vf = elems[k][0,:]
            x = (Vf + V_list[k]).argmin()
            min_x_list.append(x)
            if k < len(u_list):
                u = u_list[k][x]
                min_u_list.append(u)

        return min_u_list, min_x_list

    def parFwdBwdPass(self,x0,u_list,V_list):
        """ Run parallel forward-backward pass.

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        elems = self.parFwdBwdPass_init(x0)
        elems = par_fwdbwd_pass_scan(elems)
        min_u_list, min_x_list = self.parFwdBwdPass_extract(elems,u_list,V_list)

        return min_u_list, min_x_list

    ###########################################################################
    # Parallel forward pass with value function composition with auxiliary
    # intialization for x0
    ###########################################################################

    def parFwdBwdPass2_init(self):
        """ Initialize parallel forward-backward pass forward (v2) elements.

        Returns:
            elems: List of elements to be used in parallel value function forward pass.
        """
        T = len(self.f)
        xdim = self.f[0].shape[0]
        udim = self.f[0].shape[1]

        elems = []

        for k in range(T):
            V = np.full((xdim,xdim), _FSC_NP_INFTY)
            L = self.L[k]
            f = self.f[k]
            for x in range(xdim):
                for u in range(udim):
                    v  = L[x,u]
                    xp = f[x,u]
                    if v < V[x,xp]:
                        V[x,xp] = v
            elems.append(V)

        return elems

    def parFwdBwdPass2_extract(self,x0,elems,u_list,V_list):
        """ Extract results from parallel forward-backward pass (v2) elements.

        Parameters:
            x0: Initial state.
            elems: List of elements from the parallel forward-backward pass.
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        min_u_list = []
        min_x_list = []

        for k in range(len(elems)+1):
            if k == 0:
                x = x0
            else:
                Vf = elems[k-1][x0,:]
                x = (Vf + V_list[k]).argmin()
            min_x_list.append(x)
            if k < len(u_list):
                u = u_list[k][x]
                min_u_list.append(u)

        return min_u_list, min_x_list

    def parFwdBwdPass2(self,x0,u_list,V_list):
        """ Run parallel forward-backward pass (v2).

        Parameters:
            x0: Initial state.
            u_list: List of optimal actions vectors for k=0,...,T-1.
            V_list: List of optimal value functions for k=0,...,T.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
        """
        elems = self.parFwdBwdPass2_init()
        elems = par_fwdbwd_pass_scan(elems)
        min_u_list, min_x_list = self.parFwdBwdPass2_extract(x0,elems,u_list,V_list)

        return min_u_list, min_x_list

    ###########################################################################
    # Batch solution to the problem
    ###########################################################################

    def batch_solution(self,x0):
        """ Compute batch solution to the problem.

        Parameters:
            x0: Initial state.

        Returns:
            min_u_list: List of optimal controls for k=0,...,T-1.
            min_x_list: List of optimal states for k=0,...,T.
            min_cost: Minimum cost.
        """

        T = len(self.f)

        u_list     = [0] * T
        min_u_list = u_list.copy()
        min_x_list = []
        min_cost   = 1e10

        udim = self.f[0].shape[1]
        done = False

        while not done:
            cost = 0.0
            x = x0
            x_list = [x]
            for k in range(T):
                u = u_list[k]
                L = self.L[k]
                f = self.f[k]
                cost += L[x,u]
                x = f[x,u]
                x_list.append(x)

            cost += self.LT[x]
            if cost < min_cost:
                min_u_list = u_list.copy()
                min_x_list = x_list.copy()
                min_cost = cost

            i = 0
            while i < len(u_list) and u_list[i] == udim-1:
                u_list[i] = 0
                i = i + 1

            if i < len(u_list):
                u_list[i] += 1
            else:
                done = True

        return min_u_list, min_x_list, min_cost

    ###########################################################################
    # Cost computation
    ###########################################################################

    def cost(self, x_list, u_list):
        """ Compute cost of a given trajectory.

        Parameters:
            x_list: List of states.
            u_list: List of controls.

        Returns:
            res: Cost of the trajectory.
        """
        xT = x_list[-1]
        res = self.LT[xT]
        for k in range(len(u_list)):
            res += self.L[k][x_list[k],u_list[k]]

        return res


