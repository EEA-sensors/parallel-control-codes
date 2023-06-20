"""
Numpy version of partial condensing for LQT. Note: This implementation assumes
that M = 0, s = 0, and Z = I, otherwise it will just silently
produce a wrong result. Could be extended later though.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.lqt_np as lqt_np

##############################################################################
#
# Partial condensing for LQT
#
##############################################################################

class LQTCondenser:
    """
    LQT Condenser class
    """

    def __init__(self):
        """ Constructor. """
        self.xdim = []
        self.udim = []
        self.Nc = []
        self.clqt = []
        self.Lambda_list = []
        self.cbar_list = []
        self.Lbar_list = []

    def condense(self, lqt, Nc):
        """ Partially condense a given LQT.

        Parameters:
            lqt: LQT to be condensed.
            Nc: Number of condensed steps.

        Returns:
            clqt: Condensed LQT.
        """
        xdim = lqt.F[0].shape[0]
        udim = lqt.L[0].shape[1]
        rdim = lqt.H[0].shape[0]

        F = lqt.F[:]
        c = lqt.c[:]
        L = lqt.L[:]
        H = lqt.H[:]
        r = lqt.r[:]
        X = lqt.X[:]
        U = lqt.U[:]

        T = len(F)
        self.T = T

        # Append trivial dynamics if needed
        if (T % Nc) != 0:
            for _ in range(Nc - (T % Nc)):
                F.append(np.eye(xdim))
                c.append(np.zeros(xdim))
                L.append(np.zeros((xdim,udim)))
                H.append(np.zeros((rdim,xdim)))
                r.append(np.zeros(rdim))
                X.append(np.zeros((rdim,rdim)))
                U.append(np.eye(udim))

        T = len(F)
        Tc = T // Nc

        assert Tc * Nc == T, 'T is not divisible by Nc'

        self.Lambda_list = []
        self.cbar_list = []
        self.Lbar_list = []
        self.trans_list = []

        clqt_F = []
        clqt_c = []
        clqt_L = []
        clqt_H = []
        clqt_r = []
        clqt_X = []
        clqt_U = []
        clqt_M = []
        clqt_s = []
        clqt_HT = lqt.HT
        clqt_rT = lqt.rT
        clqt_XT = lqt.XT

        for kp in range(Tc):
            k = Nc * kp
            XUbar = np.zeros(((rdim+udim)*Nc,(rdim+udim)*Nc))
            rsbar = np.zeros((rdim+udim)*Nc)
            Hbar  = np.zeros((rdim*Nc,xdim*Nc))
            Lambda = np.zeros((xdim*Nc,xdim))
            Ftmp = np.eye(xdim)
            for i in range(0,Nc):
                XUbar[(i*rdim):((i+1)*rdim),(i*rdim):((i+1)*rdim)] = X[k + i]
                XUbar[(rdim*Nc+i*udim):(rdim*Nc+(i+1)*udim),(rdim*Nc+i*udim):(rdim*Nc+(i+1)*udim)] = U[k + i]
                rsbar[(i*rdim):((i+1)*rdim)] = r[k + i]
                Hbar[(i*rdim):((i+1)*rdim),(i*xdim):((i+1)*xdim)] = H[k + i]
                Lambda[(i*xdim):((i+1)*xdim),:] = Ftmp
                Ftmp = F[k + i] @ Ftmp

            Fstar = Ftmp

            cbar = np.zeros(xdim*Nc)
            Lbar = np.zeros((xdim*Nc,udim*Nc))
            cstar = np.zeros(xdim)
            Lstar = np.zeros((xdim,udim*Nc))
            Fpows = np.zeros((xdim,xdim*Nc))
            for i in range(1,Nc+1):
                Fpows[:, ((i-1) * xdim):(i * xdim)] = np.eye(xdim)
                for j in range(0,i-1):
                    Fpows[:, (j * xdim):((j+1) * xdim)] = \
                        F[k+i-1] @ Fpows[:, (j * xdim):((j+1) * xdim)]
                if i < Nc:
                    for j in range(0,i):
                        Lbar[(i * xdim):((i+1) * xdim), (j * udim):((j + 1) * udim)] = \
                            Fpows[:, (j * xdim):((j + 1) * xdim)] @ L[k + j]
                        cbar[(i * xdim):((i+1) * xdim)] += \
                            Fpows[:, (j * xdim):((j + 1) * xdim)] @ c[k + j]
                else:
                    for j in range(0, i):
                        Lstar[:, (j * udim):((j + 1) * udim)] = \
                            Fpows[:, (j * xdim):((j + 1) * xdim)] @ L[k + j]
                        cstar += Fpows[:, (j * xdim):((j + 1) * xdim)] @ c[k + j]

            trans = np.zeros(((rdim+udim)*Nc,xdim+udim*Nc))
            trans[0:(rdim*Nc),0:xdim] = Hbar @ Lambda
            trans[0:(rdim*Nc),xdim:] = Hbar @ Lbar
            trans[(rdim*Nc):,xdim:] = np.eye(udim*Nc)

            XUMstar = trans.T @ XUbar @ trans

            tmp = np.zeros((rdim+udim)*Nc)
            tmp[0:(rdim*Nc)] = Hbar @ cbar
            rsstar = linalg.solve(XUMstar, trans.T @ XUbar @ (rsbar - tmp), assume_a='pos')

            Xstar = XUMstar[0:xdim,0:xdim]
            Mstar = XUMstar[0:xdim,xdim:]
            Ustar = XUMstar[xdim:,xdim:]
            rstar = rsstar[0:xdim]
            sstar = rsstar[xdim:]
            Hstar = np.eye(xdim)

            clqt_F.append(Fstar)
            clqt_c.append(cstar)
            clqt_L.append(Lstar)
            clqt_H.append(Hstar)
            clqt_r.append(rstar)
            clqt_X.append(Xstar)
            clqt_U.append(Ustar)
            clqt_M.append(Mstar)
            clqt_s.append(sstar)

            self.Lambda_list.append(Lambda)
            self.Lbar_list.append(Lbar)
            self.cbar_list.append(cbar)
            self.trans_list.append(trans)

        clqt = lqt_np.LQT.checkAndExpand(clqt_F,clqt_L,clqt_X,clqt_U,clqt_XT,clqt_c,clqt_H,clqt_r,clqt_HT,clqt_rT,None,clqt_s,clqt_M,T)

        self.clqt = clqt
        self.Nc = Nc
        self.xdim = xdim
        self.udim = udim

        return clqt

    def convertUX(self, clqt_u_list, clqt_x_list):
        """ Convert condensed controls and states to uncondensed controls and states.

        Parameters:
            clqt_u_list: List of condensed controls
            clqt_x_list: List of condensed states

        Returns:
            u_list: List of uncondensed controls
            x_list: List of uncondensed states
        """
        udim = self.udim
        xdim = self.xdim
        Nc = self.Nc
        T  = self.T
        u_list = []
        x_list = []
        for i in range(len(clqt_u_list)):
            u = clqt_u_list[i]
            x = clqt_x_list[i]
            x = self.Lambda_list[i] @ x + self.cbar_list[i] + self.Lbar_list[i] @ u
            for j in range(Nc):
                if len(u_list) < T:
                    u_list.append(u[(j * udim):((j+1)*udim)])
                    x_list.append(x[(j * xdim):((j+1)*xdim)])
        x_list.append(clqt_x_list[-1])

        return u_list, x_list

