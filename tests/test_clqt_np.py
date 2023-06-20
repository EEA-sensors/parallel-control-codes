"""
Unit tests for continuous-time numpy-based Linear Quadratic Regulator and Tracker routines.

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
# Unit tests for CLQR
##############################################################################

class CLQR_np_UnitTest(unittest.TestCase):
    """Unit tests for CLQR"""

    def test_pack(self):
        rng = np.random.default_rng(123)

        n = 5
        S1 = rng.standard_normal((n, n))
        v1 = rng.standard_normal((n,))

        x = clqt_np.pack_Sv(S1, v1)
        S2, v2 = clqt_np.unpack_Sv(x)

        self.assertTrue(np.linalg.norm(S1 - S2) < 1e-10)
        self.assertTrue(np.linalg.norm(v1 - v2) < 1e-10)

        A1 = rng.standard_normal((n, n))
        b1 = rng.standard_normal((n,))
        C1 = rng.standard_normal((n, n))

        x = clqt_np.pack_abc(A1, b1, C1)
        A2, b2, C2 = clqt_np.unpack_abc(x)

        self.assertTrue(np.linalg.norm(A1 - A2) < 1e-10)
        self.assertTrue(np.linalg.norm(b1 - b2) < 1e-10)
        self.assertTrue(np.linalg.norm(C1 - C2) < 1e-10)

        Psi1 = rng.standard_normal((n, n))
        phi1 = rng.standard_normal((n,))

        x = clqt_np.pack_Psiphi(Psi1, phi1)
        Psi2, phi2 = clqt_np.unpack_Psiphi(x)

        self.assertTrue(np.linalg.norm(Psi1 - Psi2) < 1e-10)
        self.assertTrue(np.linalg.norm(phi1 - phi2) < 1e-10)

        A1 = rng.standard_normal((n, n))
        b1 = rng.standard_normal((n,))
        C1 = rng.standard_normal((n, n))
        eta1 = rng.standard_normal((n,))
        J1 = rng.standard_normal((n, n))

        x = clqt_np.pack_abcej(A1, b1, C1, eta1, J1)
        A2, b2, C2, eta2, J2 = clqt_np.unpack_abcej(x)

        self.assertTrue(np.linalg.norm(A1 - A2) < 1e-10)
        self.assertTrue(np.linalg.norm(b1 - b2) < 1e-10)
        self.assertTrue(np.linalg.norm(C1 - C2) < 1e-10)
        self.assertTrue(np.linalg.norm(eta1 - eta2) < 1e-10)
        self.assertTrue(np.linalg.norm(J1 - J2) < 1e-10)


    def test_seqBackwardPass_1(self):
        # This is a closed form solution from Lewis & Syrmos, 1995 pages 174-
        a = -0.1
        b = 1
        sT = 2
        q = 1
        r = 0.2
        T = 2

        beta = math.sqrt(a**2 + (b**2) * q / r)
        s1 = (r / b**2) * (beta - a)
        s2 = (r / b**2) * (beta + a)

        S_ref = lambda t: s2 + (s1 + s2) / (((sT + s1) / (sT - s2)) * math.exp(2*beta*(T-t)) - 1)
        K_ref = lambda t: b * S_ref(t) / r

        F = np.array([[a]])
        L = np.array([[b]])
        X = np.array([[q]])
        U = np.array([[r]])
        XT = np.array([[sT]])

        clqr = clqt_np.CLQR(F, L, X, U, XT, T)

        t = 1
        steps = 200
        t_list = np.linspace(0, T, steps+1)
        i = np.argmin(np.abs(t - t_list))

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='exact')
        self.assertTrue(linalg.norm(S_list[i][0,0] - S_ref(t)) < 1e-5)
        self.assertTrue(linalg.norm(Kx_list[i][0,0] - K_ref(t)) < 1e-5)

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='rk4')
        self.assertTrue(linalg.norm(S_list[i][0,0] - S_ref(t)) < 1e-5)
        self.assertTrue(linalg.norm(Kx_list[i][0,0] - K_ref(t)) < 1e-5)

        Kx_list_ref = [K_ref(t_list[i]) for i in range(steps)]
        S_list_ref = [S_ref(t_list[i]) for i in range(steps+1)]

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list, Kx_list_ref)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list, S_list_ref)])
        self.assertTrue(err < 1e-5)

#        plt.plot(t_list[:steps],Kx_list_ref)
#        plt.plot(t_list[:steps],[tmp[0,0] for tmp in Kx_list],'--')
#        plt.show()

    def test_seqBackwardPass_2(self):
        # This is from lqr() in Matlab:
        # [K,S] = lqr([0 1 0; 0 0 1; 0 0 -0.1],[0 0; 1 0; 0 1],eye(3),2*eye(2))
        #
        # K =
        #
        #   0.669741372639077   1.213445644222881   0.605832001296294
        #   0.226818195424275   0.605832001296293   1.063886415763117
        #
        #
        # S =
        #
        #   1.900216945298089   1.339482745278155   0.453636390848552
        #   1.339482745278155   2.426891288445764   1.211664002592588
        #   0.453636390848552   1.211664002592588   2.127772831526234
        K_ref = np.array([[0.669741372639077,1.213445644222881,0.605832001296294],
                          [0.226818195424275,0.605832001296293,1.063886415763117]])
        S_ref = np.array([[1.900216945298089,1.339482745278155,0.453636390848552],
                          [1.339482745278155,2.426891288445764,1.211664002592588],
                          [0.453636390848552,1.211664002592588,2.127772831526234]])

        F = np.array([[0,1,0],[0,0,1],[0,0,-0.1]])
        L = np.array([[0,0],[1,0],[0,1]])
        X = np.eye(3)
        U = 2 * np.eye(2)
        XT = X

        T = 100
        clqr = clqt_np.CLQR(F, L, X, U, XT, T)
        steps = 10000

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='exact')
        self.assertTrue(linalg.norm(S_list[0] - S_ref) < 1e-5)
        self.assertTrue(linalg.norm(Kx_list[0] - K_ref) < 1e-5)

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='rk4')
        self.assertTrue(linalg.norm(S_list[0] - S_ref) < 1e-5)
        self.assertTrue(linalg.norm(Kx_list[0] - K_ref) < 1e-5)

    def getSimpleCLQR(self,T=1.0):
        F = np.array([[0,1,0],[0,0,1],[0,0,-0.1]])
        L = np.array([[0,0],[1,0],[0,1]])
        X = 0.01 * np.eye(3)
        U = 0.02 * np.eye(2)
        XT = X
        return clqt_np.CLQR(F, L, X, U, XT, T)

    def test_seqBackwardPass_3(self):
        clqr = self.getSimpleCLQR()

        steps = 100
        Kx_list1, S_list1 = clqr.seqBackwardPass(steps, method='rk4')
        Kx_list2, S_list2 = clqr.seqBackwardPass(steps, method='exact')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

        Kx_list1, S_list1 = clqr.seqBackwardPass(steps, S=2*clqr.XT, method='rk4')
        Kx_list2, S_list2 = clqr.seqBackwardPass(steps, S=2*clqr.XT, method='exact')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)


    def test_seqForwardPass_1(self):
        # Test with
        #       u = -k x
        #   dx/dt = c x + l u
        #
        # i.e.
        #   x(t) = exp((c - l k) t) x0
        #   u(t) = -k x(t)

        c = -0.5
        l = 1
        k = 0.2
        x0 = 1
        T = 1
        steps = 10000

        ts = np.linspace(0,T,steps+1)
        x_list_ref = x0 * np.exp((c - l*k) * ts)
        u_list_ref = -k * x_list_ref[:-1]

        Kx_list = steps * [k*np.eye(1)]
        clqr = clqt_np.CLQR(c*np.eye(1), l*np.eye(1), 1*np.eye(1), 1*np.eye(1), 1*np.eye(1), T)

        u_list, x_list = clqr.seqForwardPass(np.array([x0]), Kx_list, method='rk4', u_zoh=False)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list, u_list_ref)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list, x_list_ref)])
        self.assertTrue(err < 1e-5)

        u_list, x_list = clqr.seqForwardPass(np.array([x0]), Kx_list, method='exact', u_zoh=False)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list, u_list_ref)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list, x_list_ref)])
        self.assertTrue(err < 1e-5)

        u_list, x_list = clqr.seqForwardPass(np.array([x0]), Kx_list, method='rk4', u_zoh=True)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list, u_list_ref)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list, x_list_ref)])
        self.assertTrue(err < 1e-5)

        u_list, x_list = clqr.seqForwardPass(np.array([x0]), Kx_list, method='rk4', u_zoh=True)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list, u_list_ref)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list, x_list_ref)])
        self.assertTrue(err < 1e-5)

    def test_seqForwardPass_2(self):
        # Simple cross check of all the methods

        clqr = self.getSimpleCLQR(10.0)
        steps = 10000

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='exact')

        x0 = np.array([1,1,1])
        u_list1, x_list1 = clqr.seqForwardPass(x0, Kx_list, method='rk4', u_zoh=False)
        u_list2, x_list2 = clqr.seqForwardPass(x0, Kx_list, method='rk4', u_zoh=True)
        u_list3, x_list3 = clqr.seqForwardPass(x0, Kx_list, method='exact', u_zoh=False)
        u_list4, x_list4 = clqr.seqForwardPass(x0, Kx_list, method='exact', u_zoh=True)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        print(err)
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-3)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list3)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list3)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list2, u_list4)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list2, x_list4)])
        self.assertTrue(err < 1e-5)

    def test_seqFwdBwdPass_1(self):
        clqr = self.getSimpleCLQR()
        steps = 1000

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='rk4')

        x0 = np.array([1,1,1])
        u_list1, x_list1 = clqr.seqForwardPass(x0, Kx_list, method='rk4', u_zoh=False)

        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, steps)
        u_list2, x_list2 = clqr.combineForwardBackward(Kx_list, S_list, A_list, b_list, C_list)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-3)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-3)

    def test_parBackwardPass_init_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 1
        steps = 100

        Kx_list, S_list = clqr.seqBackwardPass(steps, method='rk4')

        elems1 = clqr.parBackwardPass_init(blocks, steps, forward=False)
        A1, b1, C1, eta1, J1 = lqt_np.combine_abcej(*elems1[0],*elems1[1])

        err = linalg.norm(S_list[0] - J1)
        self.assertTrue(err < 1e-5)

        elems2 = clqr.parBackwardPass_init(blocks, steps, forward=True)
        A2, b2, C2, eta2, J2 = lqt_np.combine_abcej(*elems2[0],*elems2[1])

        err = linalg.norm(S_list[0] - J2)
        self.assertTrue(err < 1e-5)

        for i in range(len(elems1)):
            for j in range(len(elems1[i])):
                err = linalg.norm(elems1[i][j] - elems2[i][j])
                self.assertTrue(err < 1e-5)

    def test_parBackwardPass_init_2(self):
        clqr = self.getSimpleCLQR()

        blocks = 100
        steps = 10

        elems = clqr.parBackwardPass_init(blocks, steps)
        elems = lqt_np.par_backward_pass_scan(elems)

        (A,b,C,eta,J) = elems[0]
        S_list = [J]
        for k in range(len(elems)-1):
            (A,b,C,eta,J) = elems[k+1]
            S_list.append(J)

        steps2 = blocks * steps
        Kx_list2, S_list2 = clqr.seqBackwardPass(steps2, method='exact')
        S_list2 = S_list2[0::steps]

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list, S_list2)])
        self.assertTrue(err < 1e-5)


    def test_parBackwardPass_extract_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 1
        steps = 10

        elems = clqr.parBackwardPass_init(blocks, steps)

        e = lqt_np.combine_abcej(*elems[0],*elems[1])

        A1, b1, C1, eta1, J1 = e
        A2, b2, C2, eta2, J2 = elems[1]

        Kx_list1, S_list1 = clqr.parBackwardPass_extract([e,elems[1]], steps, method='rk4')
        Kx_list2, S_list2 = clqr.seqBackwardPass(steps, method='rk4')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

        Kx_list1, S_list1 = clqr.parBackwardPass_extract([e,elems[1]], steps, method='exact')
        Kx_list2, S_list2 = clqr.seqBackwardPass(steps, method='exact')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_extract_2(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 2
        steps = 10

        # Initialize
        elems = clqr.parBackwardPass_init(blocks, steps, method='rk4')

        # Call the associative scan
        elems = lqt_np.par_backward_pass_scan(elems)

        # Extract the results
        Kx_list1, S_list1 = clqr.parBackwardPass_extract(elems, steps, method='rk4')
        Kx_list2, S_list2 = clqr.parBackwardPass_extract(elems, steps, method='exact')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)


    def test_parBackwardPass_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 10
        steps = 10

        Kx_list1, S_list1 = clqr.parBackwardPass(blocks, steps)
        Kx_list2, S_list2 = clqr.seqBackwardPass(blocks * steps, method='rk4')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

        Kx_list1, S_list1 = clqr.parBackwardPass(blocks, steps, extract_method='exact')
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

    def test_parForwardPass_init_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 1
        steps = 100

        x0 = np.array([1,2,1])
        Kx_list, S_list = clqr.seqBackwardPass(steps, method='rk4')
        u_list, x_list = clqr.seqForwardPass(x0, Kx_list, method='rk4', u_zoh=False)

        elems1 = clqr.parForwardPass_init(x0, Kx_list, blocks, steps, forward=True)
        Psi1, phi1 = lqt_np.combine_fc(*elems1[0],*elems1[1])

        err = linalg.norm(x_list[-1] - phi1)
        self.assertTrue(err < 1e-5)

        elems2 = clqr.parForwardPass_init(x0, Kx_list, blocks, steps, forward=False)
        Psi2, phi2 = lqt_np.combine_fc(*elems2[0],*elems2[1])

        err = linalg.norm(x_list[-1] - phi2)
        self.assertTrue(err < 1e-5)

        for i in range(len(elems1)):
            for j in range(len(elems1[i])):
                err = linalg.norm(elems1[i][j] - elems2[i][j])
                self.assertTrue(err < 1e-5)

    def test_parForwardPass_extract_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 2
        steps = 10

        x0 = np.array([1,2,1])
        Kx_list1, S_list1 = clqr.seqBackwardPass(blocks * steps, method='rk4')
        u_list1, x_list1 = clqr.seqForwardPass(x0, Kx_list1, method='rk4', u_zoh=False)

        elems = clqr.parForwardPass_init(x0, Kx_list1, blocks, steps, forward=True)
        elems = lqt_np.par_forward_pass_scan(elems)
        u_list2, x_list2 = clqr.parForwardPass_extract(Kx_list1, elems, steps, method='rk4', u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parForwardPass_1(self):
        clqr = self.getSimpleCLQR()

        blocks = 10
        steps = 100
        x0 = np.array([1,2,1])

        Kx_list1, S_list1 = clqr.seqBackwardPass(blocks * steps)
        u_list1, x_list1 = clqr.seqForwardPass(x0, Kx_list1, method='rk4', u_zoh=False)

        Kx_list2, S_list2 = clqr.parBackwardPass(blocks, steps)
        u_list2, x_list2 = clqr.parForwardPass(x0, Kx_list2, blocks, steps, init_method='rk4', extract_method='rk4', u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parForwardPass_2(self):
        clqr = self.getSimpleCLQR()

        blocks = 10
        steps = 100
        x0 = np.array([1,2,1])

        Kx_list1, S_list1 = clqr.seqBackwardPass(blocks * steps)
        Kx_list2, S_list2 = clqr.parBackwardPass(blocks, steps)

        u_list1, x_list1 = clqr.seqForwardPass(x0, Kx_list1, method='exact', u_zoh=False)
        u_list2, x_list2 = clqr.parForwardPass(x0, Kx_list2, blocks, steps, init_method='rk4', extract_method='exact', u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parFwdBwdPass_init_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 1
        steps = 100

        x0 = np.array([1,2,1])
        elems = clqr.parFwdBwdPass_init(x0, blocks, steps, method='rk4', forward=True)

        e = lqt_np.combine_abcej(*elems[0],*elems[1])

        A, b, C, eta, J = e

        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, steps)

        err = linalg.norm(b - b_list[-1])
        self.assertTrue(err < 1e-4)

        err = linalg.norm(C - C_list[-1])
        self.assertTrue(err < 1e-4)

    def test_parFwdBwdPass_init_2(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 10
        steps = 10

        x0 = np.array([1,2,1])
        elems = clqr.parFwdBwdPass_init(x0, blocks, steps, method='rk4', forward=True)

        elems = lqt_np.par_fwdbwd_pass_scan(elems)

        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, blocks * steps)

        A, b, C, eta, J = elems[-1]

        err = linalg.norm(b - b_list[-1])
        self.assertTrue(err < 1e-4)

        err = linalg.norm(C - C_list[-1])
        self.assertTrue(err < 1e-4)

    def test_parFwdBwdPass_extract_1(self):
        clqr = self.getSimpleCLQR(0.1)

        blocks = 1
        steps = 100

        Kx_list, S_list = clqr.seqBackwardPass(blocks * steps, method='rk4')

        x0 = np.array([1,1,2])
        elems = clqr.parFwdBwdPass_init(x0, blocks, steps, method='rk4', forward=True)
        elems = lqt_np.par_fwdbwd_pass_scan(elems)
        u_list1, x_list1 = clqr.parFwdBwdPass_extract(Kx_list, S_list, elems, steps, method='rk4')

        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, blocks * steps, method='rk4')
        u_list2, x_list2 = clqr.combineForwardBackward(Kx_list, S_list, A_list, b_list, C_list)

        Kx_list3, S_list3 = clqr.seqBackwardPass(blocks * steps, method='rk4')
        u_list3, x_list3 = clqr.seqForwardPass(x0, Kx_list3, method='rk4')

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list3)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list3)])
        self.assertTrue(err < 1e-4)

    def test_parFwdBwdPass_1(self):
        clqr = self.getSimpleCLQR()

        blocks = 10
        steps = 10

        x0 = np.array([1,1,2])

        Kx_list, S_list = clqr.seqBackwardPass(blocks * steps, method='rk4')
        u_list1, x_list1 = clqr.parFwdBwdPass(x0, Kx_list, S_list, blocks, steps)

        Kx_list2, S_list2 = clqr.seqBackwardPass(blocks * steps, method='rk4')
        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, blocks * steps, method='rk4')
        u_list2, x_list2 = clqr.combineForwardBackward(Kx_list2, S_list2, A_list, b_list, C_list)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parFwdBwdPass_2(self):
        clqr = self.getSimpleCLQR()

        blocks = 10
        steps = 10

        x0 = np.array([1,1,2])

        Kx_list1, S_list1 = clqr.parBackwardPass(blocks, steps)
        u_list1, x_list1 = clqr.parFwdBwdPass(x0, Kx_list1, S_list1, blocks, steps)

        Kx_list2, S_list2 = clqr.seqBackwardPass(blocks * steps, method='rk4')
        A_list, b_list, C_list = clqr.seqFwdBwdPass(x0, blocks * steps, method='rk4')
        u_list2, x_list2 = clqr.combineForwardBackward(Kx_list2, S_list2, A_list, b_list, C_list)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)



##############################################################################
# Unit tests for CLQT
##############################################################################

class CLQT_np_UnitTest(unittest.TestCase):
    """Unit tests for CLQT"""

    def getSimpleCLQR(self,T=1.0):
        F = np.array([[0,1,0],[0,0,1],[0,0,-0.1]])
        L = np.array([[0,0],[1,0],[0,1]])
        X = 0.01 * np.eye(3)
        U = 0.02 * np.eye(2)
        XT = X
        return clqt_np.CLQR(F, L, X, U, XT, T)

    def getSimpleCLQT(self,T=1.0):
        clqr = self.getSimpleCLQR(T)
        c = np.zeros((3,))
        H = np.eye(3)
        r = np.zeros((3,))
        HT = np.eye(3)
        rT = np.zeros((3,))
        clqt = clqt_np.CLQT(lambda t: clqr.F, lambda t: clqr.L, lambda t: clqr.X, lambda t: clqr.U, clqr.XT,
                            lambda t: c, lambda t: H, lambda t: r, HT, rT, T)
        return clqt, clqr

    def test_seqBackwardPass_1(self):
        clqt, clqr = self.getSimpleCLQT()

        steps = 10
        Kx_list1, S_list1 = clqr.seqBackwardPass(steps, method='rk4')
        d_list1 = steps * np.zeros((2,))
        v_list1 = (steps+1) * np.zeros((3,))
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-5)

    def getRandomCLQT(self,T=10.0):
        X = lambda t: (0.02 + 0.01 * np.sin(1.0 * t)) * np.eye(2)
        U = lambda t: (0.04 + 0.01 * np.cos(0.3 * t)) * np.eye(1)
        XT = 0.02 * np.eye(2)
        F = lambda t: np.array([[0, 1], [-(2.0 * np.sin(3.1 * t)) ** 2, 0]])
        L = lambda t: np.array([[0], [1 + 0.1 * np.sin(2.1 * t)]])
        HT = np.array([[1.0,0.1],[0.0,1.0]])
        H = lambda t: np.array([[1.0,0.0],[0.1,1.0]])
        c = lambda t: np.array([0,np.cos(0.98 * t)])
        r = lambda t: np.array([np.cos(0.5 * t), np.sin(0.5 * t)])
        rT = np.array([0.1, 0])

        clqt = clqt_np.CLQT(F, L, X, U, XT, c, H, r, HT, rT, T)
        t = 1.0
        return clqt

    def getDiscreteLQT(self, clqt, steps, t0=0.0):
        dt = clqt.T / steps
        t_list = np.arange(steps) * dt + t0

        I = np.eye((clqt.F(0)).shape[0])
        dlqt = lqt_np.LQT.checkAndExpand([I + clqt.F(t) * dt for t in t_list],
                                         [clqt.L(t) * dt for t in t_list],
                                         [clqt.X(t) * dt for t in t_list],
                                         [clqt.U(t) * dt for t in t_list],
                                         clqt.XT,
                                         [clqt.c(t) * dt for t in t_list],
                                         [clqt.H(t) for t in t_list],
                                         [clqt.r(t) for t in t_list],
                                         clqt.HT, clqt.rT)
        return dlqt

    def test_seqBackwardPass_2(self):
        # Generate (random) model and check with dense discretization
        # for discrete LQT that the result matches
        clqt = self.getRandomCLQT()

        steps = 10000
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(steps)

        dlqt = self.getDiscreteLQT(clqt, steps)
        Kx_list2, d_list2, S_list2, v_list2 = dlqt.seqBackwardPass()

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-2)

#        dt = clqt.T / steps
#        t = np.arange(steps) * dt
#        plt.plot(t,[tmp[0,0] for tmp in Kx_list1],'r')
#        plt.plot(t,[tmp[0,0] for tmp in Kx_list2],'k--')
#        plt.show()

    def test_seqForwardPass_1(self):
        clqt, clqr = self.getSimpleCLQT()

        steps = 10
        Kx_list1, S_list1 = clqr.seqBackwardPass(steps, method='rk4')
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        x0 = np.array([1,2,3])
        u_list1a, x_list1a = clqr.seqForwardPass(x0, Kx_list1, method='rk4', u_zoh=False)
        u_list1b, x_list1b = clqr.seqForwardPass(x0, Kx_list1, method='rk4', u_zoh=True)

        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=False)
        u_list2b, x_list2b = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=True)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1a, u_list2a)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1a, x_list2a)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1b, u_list2b)])
        self.assertTrue(err < 1e-5)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1b, x_list2b)])
        self.assertTrue(err < 1e-5)

    def test_seqForwardPass_2(self):
        # Generate (random) model and check with dense discretization
        # for discrete LQT that the result matches
        clqt = self.getRandomCLQT()

        steps = 10000
        dlqt = self.getDiscreteLQT(clqt, steps)
        Kx_list1, d_list1, S_list1, v_list1 = dlqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        x0 = np.array([1,2])

        u_list1, x_list1 = dlqt.seqForwardPass(x0, Kx_list1, d_list1)

        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=False)
        u_list2b, x_list2b = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=True)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2a)])
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2a)])
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2b)])
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2b)])
        self.assertTrue(err < 1e-2)

    def test_seqFwdBwdPass_1(self):
        clqt, clqr = self.getSimpleCLQT()

        steps = 10
        x0 = np.array([1,2,3])

        Kx_list1, S_list1 = clqr.seqBackwardPass(steps, method='rk4')
        A_list1, b_list1, C_list1 = clqr.seqFwdBwdPass(x0, steps)
        u_list1, x_list1 = clqr.combineForwardBackward(Kx_list1, S_list1, A_list1, b_list1, C_list1)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)
        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps)
        u_list2, x_list2 = clqt.combineForwardBackward(Kx_list2, d_list2, S_list2, v_list2, A_list2, b_list2, C_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-3)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-3)

    def test_seqFwdBwdPass_2(self):
        # Generate random model and check that the two methods match
        clqt = self.getRandomCLQT()

        steps = 1000
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(steps)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        x0 = np.array([1,2])
        A_list1, b_list1, C_list1 = clqt.seqFwdBwdPass(x0, steps)
        u_list1, x_list1 = clqt.combineForwardBackward(Kx_list1, d_list1, S_list1, v_list1, A_list1, b_list1, C_list1)

        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=False)
        u_list2b, x_list2b = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=True)

#        dt = clqt.T / steps
#        t = np.arange(steps) * dt
#        plt.plot(t, [tmp[0] for tmp in x_list1[1:]], 'r')
#        plt.plot(t, [tmp[0] for tmp in x_list2a[1:]], 'k--')
#        plt.show()

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2a)])
        print(err)
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2a)])
        print(err)
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2b)])
        print(err)
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2b)])
        print(err)
        self.assertTrue(err < 1e-2)


    def test_parBackwardPass_init_1(self):
        clqt, _ = self.getSimpleCLQT(0.1)

        blocks = 1
        steps = 100

        Kx_list, d_list, S_list, v_list = clqt.seqBackwardPass(steps)

        elems1 = clqt.parBackwardPass_init(blocks, steps, forward=False)
        A1, b1, C1, eta1, J1 = lqt_np.combine_abcej(*elems1[0],*elems1[1])

        err = linalg.norm(S_list[0] - J1)
        self.assertTrue(err < 1e-5)

        elems2 = clqt.parBackwardPass_init(blocks, steps, forward=True)
        A2, b2, C2, eta2, J2 = lqt_np.combine_abcej(*elems2[0],*elems2[1])

        err = linalg.norm(S_list[0] - J2)
        self.assertTrue(err < 1e-5)

        for i in range(len(elems1)):
            for j in range(len(elems1[i])):
                err = linalg.norm(elems1[i][j] - elems2[i][j])
                self.assertTrue(err < 1e-5)

    def test_parBackwardPass_init_2(self):
        clqt, _ = self.getSimpleCLQT(0.1)

        blocks = 100
        steps = 10

        elems = clqt.parBackwardPass_init(blocks, steps)
        elems = lqt_np.par_backward_pass_scan(elems)

        (A,b,C,eta,J) = elems[0]
        S_list = [J]
        for k in range(len(elems)-1):
            (A,b,C,eta,J) = elems[k+1]
            S_list.append(J)

        steps2 = blocks * steps
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps2)
        S_list2 = S_list2[0::steps]

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list, S_list2)])
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_extract_1(self):
        clqt, _ = self.getSimpleCLQT(0.1)

        blocks = 1
        steps = 10

        elems = clqt.parBackwardPass_init(blocks, steps)

        e = lqt_np.combine_abcej(*elems[0],*elems[1])

        A1, b1, C1, eta1, J1 = e
        A2, b2, C2, eta2, J2 = elems[1]

        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass_extract([e,elems[1]], steps)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_extract_2(self):
        clqt, clqr = self.getSimpleCLQT(1.0)

        blocks = 10
        steps = 10

        # Initialize
        elems1 = clqt.parBackwardPass_init(blocks, steps)
        elems2 = clqr.parBackwardPass_init(blocks, steps, method='rk4')

        # Call the associative scan
        elems1 = lqt_np.par_backward_pass_scan(elems1)
        elems2 = lqt_np.par_backward_pass_scan(elems2)

        # Extract the results
        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass_extract(elems1, steps)
        Kx_list2, S_list2 = clqr.parBackwardPass_extract(elems2, steps, method='rk4')
        d_list2 = steps * np.zeros((2,))
        v_list2 = (steps+1) * np.zeros((3,))

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-5)


    def test_parBackwardPass_1(self):
        clqt, clqr = self.getSimpleCLQT(10.0)

        blocks = 20
        steps = 15

        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass(blocks, steps)
        Kx_list2, S_list2 = clqr.parBackwardPass(blocks, steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_2(self):
        # Generate (random) model and check that the results match
        clqt = self.getRandomCLQT()

        blocks = 20
        steps = 15

        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(steps * blocks)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(S_list1, S_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(Kx_list1, Kx_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(d_list1, d_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(v_list1, v_list2)])
        self.assertTrue(err < 1e-5)

#        dt = clqt.T / steps
#        t = np.arange(steps) * dt
#        plt.plot(t,[tmp[0,0] for tmp in Kx_list1],'r')
#        plt.plot(t,[tmp[0,0] for tmp in Kx_list2],'k--')
#        plt.show()

    def test_parForwardPass_init_1(self):
        clqt, clqr = self.getSimpleCLQT(10.0)

        blocks = 1
        steps = 100

        x0 = np.array([1,2,1])
        Kx_list, d_list, S_list, v_list = clqt.seqBackwardPass(steps)
        u_list, x_list = clqt.seqForwardPass(x0, Kx_list, d_list, u_zoh=False)

        elems1 = clqt.parForwardPass_init(x0, Kx_list, d_list, blocks, steps, forward=True)
        Psi1, phi1 = lqt_np.combine_fc(*elems1[0],*elems1[1])

        err = linalg.norm(x_list[-1] - phi1)
        self.assertTrue(err < 1e-5)

        elems2 = clqt.parForwardPass_init(x0, Kx_list, d_list, blocks, steps, forward=False)
        Psi2, phi2 = lqt_np.combine_fc(*elems2[0],*elems2[1])

        err = linalg.norm(x_list[-1] - phi2)
        self.assertTrue(err < 1e-5)

        for i in range(len(elems1)):
            for j in range(len(elems1[i])):
                err = linalg.norm(elems1[i][j] - elems2[i][j])
                self.assertTrue(err < 1e-5)

    def test_parForwardPass_extract_1(self):
        clqt, clqr = self.getSimpleCLQT(10.0)

        blocks = 2
        steps = 10

        x0 = np.array([1,2,1])
        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(blocks * steps)
        u_list1, x_list1 = clqt.seqForwardPass(x0, Kx_list1, d_list1, u_zoh=False)

        elems = clqt.parForwardPass_init(x0, Kx_list1, d_list1, blocks, steps, forward=True)
        elems = lqt_np.par_forward_pass_scan(elems)
        u_list2, x_list2 = clqt.parForwardPass_extract(Kx_list1, d_list1, elems, steps, u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parForwardPass_1(self):
        clqt, clqr = self.getSimpleCLQT(10.0)

        blocks = 10
        steps = 100
        x0 = np.array([1,2,1])

        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass(blocks, steps)
        u_list1, x_list1 = clqt.parForwardPass(x0, Kx_list1, d_list1, blocks, steps, u_zoh=False)

        Kx_list2, S_list2 = clqr.parBackwardPass(blocks, steps, init_method='rk4', extract_method='rk4')
        u_list2, x_list2 = clqr.parForwardPass(x0, Kx_list2, blocks, steps, init_method='rk4', extract_method='rk4', u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-4)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-4)

    def test_parForwardPass_2(self):
        # Generate (random) model and check that the results match
        clqt = self.getRandomCLQT(20.0)

        blocks = 20
        steps = 50

        x0 = np.array([2,1])

        Kx_list1, d_list1, S_list1, v_list1 = clqt.seqBackwardPass(steps * blocks)
        u_list1, x_list1 = clqt.seqForwardPass(x0, Kx_list1, d_list1, u_zoh=False)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        u_list2, x_list2 = clqt.parForwardPass(x0, Kx_list2, d_list2, blocks, steps, u_zoh=False)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-5)

#        dt = clqt.T / steps / blocks
#        t = np.arange(steps * blocks) * dt
#        plt.plot(t,[tmp[0] for tmp in x_list1[1:]],'r')
#        plt.plot(t,[tmp[0] for tmp in x_list2[1:]],'k--')
#        plt.show()

    def test_parFwdBwdPass_1(self):
        clqt, clqr = self.getSimpleCLQT()

        blocks = 10
        steps = 10
        x0 = np.array([1,2,3])

        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass(blocks, steps)
        u_list1, x_list1 = clqt.parFwdBwdPass(x0, Kx_list1, d_list1, S_list1, v_list1, blocks, steps)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps * blocks)
        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps * blocks)
        u_list2, x_list2 = clqt.combineForwardBackward(Kx_list2, d_list2, S_list2, v_list2, A_list2, b_list2, C_list2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-5)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-5)

    def test_parFwdBwdPass_2(self):
        clqt = self.getRandomCLQT(20.0)

        blocks = 100
        steps = 100

        x0 = np.array([2,1])

        Kx_list1, d_list1, S_list1, v_list1 = clqt.parBackwardPass(blocks, steps)
        u_list1, x_list1 = clqt.parFwdBwdPass(x0, Kx_list1, d_list1, S_list1, v_list1, blocks, steps)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps * blocks)
        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps * blocks)
        u_list2, x_list2 = clqt.combineForwardBackward(Kx_list2, d_list2, S_list2, v_list2, A_list2, b_list2, C_list2)

        Kx_list3, d_list3, S_list3, v_list3 = clqt.seqBackwardPass(steps * blocks)
        u_list3, x_list3 = clqt.seqForwardPass(x0, Kx_list3, d_list3, u_zoh=False)


#        dt = clqt.T / steps / blocks
#        t = np.arange(steps * blocks) * dt
#        plt.plot(t, [tmp1[0]-tmp2[0] for tmp1, tmp2 in zip(x_list1[1:], x_list3[1:])], 'k--')
#        plt.show()

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        print(err)
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        print(err)
        self.assertTrue(err < 1e-2)

        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list3)])
        print(err)
        self.assertTrue(err < 1e-2)
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list3)])
        print(err)
        self.assertTrue(err < 1e-2)
