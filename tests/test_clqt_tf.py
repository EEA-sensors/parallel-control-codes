"""
Unit tests for TensorFlow-based continuous-time Linear Quadratic Tracker (LQT) routines.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf

import unittest
import parallel_control.clqt_np as clqt_np
import parallel_control.clqt_tf as clqt_tf

import matplotlib.pyplot as plt


class CLQT_tf_UnitTest(unittest.TestCase):

    def test_pack(self):
        rng = np.random.default_rng(123)

        n = 5
        S1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)
        v1 = tf.constant(rng.standard_normal((n,)), dtype=tf.float64)

        x = clqt_tf.pack_Sv(S1, v1)
        S2, v2 = clqt_np.unpack_Sv(x)

        self.assertTrue(tf.reduce_max(tf.math.abs(S1 - S2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(v1 - v2)) < 1e-10)

        A1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)
        b1 = tf.constant(rng.standard_normal((n,)), dtype=tf.float64)
        C1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)

        x = clqt_tf.pack_abc(A1, b1, C1)
        A2, b2, C2 = clqt_tf.unpack_abc(x)

        self.assertTrue(tf.reduce_max(tf.math.abs(A1 - A2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(b1 - b2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(C1 - C2)) < 1e-10)

        Psi1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)
        phi1 = tf.constant(rng.standard_normal((n,)), dtype=tf.float64)

        x = clqt_tf.pack_Psiphi(Psi1, phi1)
        Psi2, phi2 = clqt_tf.unpack_Psiphi(x)

        self.assertTrue(tf.reduce_max(tf.math.abs(Psi1 - Psi2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(phi1 - phi2)) < 1e-10)

        A1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)
        b1 = tf.constant(rng.standard_normal((n,)), dtype=tf.float64)
        C1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)
        eta1 = tf.constant(rng.standard_normal((n,)), dtype=tf.float64)
        J1 = tf.constant(rng.standard_normal((n, n)), dtype=tf.float64)

        x = clqt_tf.pack_abcej(A1, b1, C1, eta1, J1)
        A2, b2, C2, eta2, J2 = clqt_tf.unpack_abcej(x)

        self.assertTrue(tf.reduce_max(tf.math.abs(A1 - A2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(b1 - b2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(C1 - C2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(eta1 - eta2)) < 1e-10)
        self.assertTrue(tf.reduce_max(tf.math.abs(J1 - J2)) < 1e-10)


    def test_seqBackwardPass_1(self):
        # This is a closed form solution from Lewis & Syrmos, 1995 pages 174-
        a = tf.constant(-0.1, dtype=tf.float64)
        b = tf.constant(1, dtype=tf.float64)
        sT = tf.constant(2, dtype=tf.float64)
        q = tf.constant(1, dtype=tf.float64)
        r = tf.constant(0.2, dtype=tf.float64)
        T = tf.constant(2, dtype=tf.float64)

        beta = tf.sqrt(a**2 + (b**2) * q / r)
        s1 = (r / b**2) * (beta - a)
        s2 = (r / b**2) * (beta + a)

        S_ref = lambda t: s2 + (s1 + s2) / (((sT + s1) / (sT - s2)) * tf.exp(2.0 * beta*(T-t)) - 1.0)
        K_ref = lambda t: b * S_ref(t) / r

        F_f = lambda t: tf.constant(np.array([[a]]), dtype=tf.float64)
        L_f = lambda t: tf.constant(np.array([[b]]), dtype=tf.float64)
        X_f = lambda t: tf.constant(np.array([[q]]), dtype=tf.float64)
        U_f = lambda t: tf.constant(np.array([[r]]), dtype=tf.float64)
        c_f = lambda t: tf.zeros((1,), dtype=tf.float64)
        H_f = lambda t: tf.eye(1, dtype=tf.float64)
        r_f = lambda t: tf.zeros((1,), dtype=tf.float64)
        XT = tf.constant(np.array([[sT]]), dtype=tf.float64)
        rT = tf.zeros((1,), dtype=tf.float64)
        HT = tf.eye(1, dtype=tf.float64)

        steps = 200
        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)

        Ss, vs, Kxs, ds = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        t = 1
        t_list = np.linspace(0, T, steps+1)
        i = np.argmin(np.abs(t - t_list))

        Kxs_ref = tf.vectorized_map(K_ref, tf.reshape(t_list, shape=(tf.size(t_list),1,1)))[:steps,:,:]
        Ss_ref = tf.vectorized_map(S_ref, tf.reshape(t_list, shape=(tf.size(t_list),1,1)))

        plt.plot(t_list[:steps],Kxs_ref[:,0,0])
        plt.plot(t_list[:steps],Kxs[:,0,0],'--')
        plt.show()

        plt.plot(t_list,Ss_ref[:,0,0])
        plt.plot(t_list,Ss[:,0,0],'--')
        plt.show()

        self.assertTrue(tf.math.abs(Ss[i] - S_ref(t)) < 1e-5)
        self.assertTrue(tf.math.abs(Kxs[i] - K_ref(t)) < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss - Ss_ref))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs - Kxs_ref))
        print(err)
        self.assertTrue(err < 1e-5)

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

        T  = tf.constant(T, dtype=tf.float64)
        XT = tf.constant(clqt.XT, dtype=tf.float64)
        HT = tf.constant(clqt.HT, dtype=tf.float64)
        rT = tf.constant(clqt.rT, dtype=tf.float64)

        F_f = lambda t: tf.constant(clqr.F, dtype=tf.float64)
        L_f = lambda t: tf.constant(clqr.L, dtype=tf.float64)
        X_f = lambda t: tf.constant(clqr.X, dtype=tf.float64)
        U_f = lambda t: tf.constant(clqr.U, dtype=tf.float64)
        c_f = lambda t: tf.constant(c, dtype=tf.float64)
        H_f = lambda t: tf.constant(H, dtype=tf.float64)
        r_f = lambda t: tf.constant(r, dtype=tf.float64)

        return clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f

    def test_seqBackwardPass_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getSimpleCLQT()

        steps = 10

        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
        ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
        Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
        vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_seqBackwardPass_3(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getSimpleCLQT()

        steps = 10

        t0_list = []
        ST_list = []
        vT_list = []
        for i in range(5):
            dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
            t0 = 0.1 * i
            ST = ST + 0.01 * i * tf.eye(ST.shape[0], dtype=ST.dtype)
            vT = vT + 0.01 * i
            t0_list.append(t0)
            ST_list.append(ST)
            vT_list.append(vT)

        t0 = tf.convert_to_tensor(t0_list, dtype=tf.float64)
        ST = tf.convert_to_tensor(ST_list, dtype=tf.float64)
        vT = tf.convert_to_tensor(vT_list, dtype=tf.float64)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        print(Ss1.shape)

        for i in range(len(t0_list)):
            Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps, t0=0.1 * i,
                                                                       S=tf.make_ndarray(tf.make_tensor_proto(ST_list[i])),
                                                                       v=tf.make_ndarray(tf.make_tensor_proto(vT_list[i])))
            Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
            ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
            Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
            vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

            err = tf.reduce_max(tf.math.abs(Kxs1[i, ...] - Kxs2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(ds1[i, ...] - ds2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(Ss1[i, ...] - Ss2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(vs1[i, ...] - vs2))
            print(err)
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


        # TODO: There must be a better way than this:
        dp = lambda v: tf.constant(v, dtype=tf.float64)

        X_f = lambda t: (dp(0.02) + dp(0.01) * tf.sin(dp(1.0) * t)) * tf.eye(2, dtype=tf.float64)
        U_f = lambda t: (dp(0.04) + dp(0.01) * tf.cos(dp(0.3) * t)) * tf.eye(1, dtype=tf.float64)
        XT = tf.constant(XT, dtype=tf.float64)
        F_f = lambda t: tf.stack([tf.stack([dp(0.0), dp(1.0)]), tf.stack([-(dp(2.0) * tf.sin(dp(3.1) * t)) ** 2, dp(0.0)])])
#        L_f = lambda t: tf.stack([tf.stack([dp(0.0)]), tf.stack([dp(1.0) + dp(0.1) * tf.sin(dp(2.1) * t)])])
        def L_f(t):
            ta = tf.TensorArray(tf.float64, size=2, dynamic_size=False, infer_shape=True)
            ta = ta.write(0, tf.constant([0.0], dtype=tf.float64))
            ta = ta.write(1, tf.expand_dims(dp(1.0) + 0.1 * tf.sin(2.1 * t), -1))
            return ta.stack()

        HT = tf.constant(HT, dtype=tf.float64)
        H_f = lambda t: tf.stack([tf.stack([dp(1.0),dp(0.0)]), tf.stack([dp(0.1),dp(1.0)])])
        c_f = lambda t: tf.stack([dp(0.0), tf.cos(dp(0.98) * t)])
        r_f = lambda t: tf.stack([tf.cos(dp(0.5) * t), tf.sin(dp(0.5) * t)])
        rT = tf.constant(rT, dtype=tf.float64)
        T  = tf.constant(clqt.T, dtype=tf.float64)

        return clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f


    def test_seqBackwardPass_4(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)

        Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
        ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
        Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
        vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_seqBackwardPass_5(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        t0_list = []
        ST_list = []
        vT_list = []
        for i in range(5):
            dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
            t0 = 0.1 * i
            ST = ST + 0.01 * i * tf.eye(ST.shape[0], dtype=ST.dtype)
            vT = vT + 0.01 * i
            t0_list.append(t0)
            ST_list.append(ST)
            vT_list.append(vT)

        t0 = tf.convert_to_tensor(t0_list, dtype=tf.float64)
        ST = tf.convert_to_tensor(ST_list, dtype=tf.float64)
        vT = tf.convert_to_tensor(vT_list, dtype=tf.float64)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        print(Ss1.shape)
        print(t0)
        print(ST)
        print(vT)

        for i in range(len(t0_list)):
            Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps, t0=t0_list[i],
                                                                       S=ST_list[i].numpy(),
                                                                       v=vT_list[i].numpy())
            Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
            ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
            Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
            vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

            err = tf.reduce_max(tf.math.abs(Kxs1[i, ...] - Kxs2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(ds1[i, ...] - ds2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(Ss1[i, ...] - Ss2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(vs1[i, ...] - vs2))
            print(err)
            self.assertTrue(err < 1e-5)

    def test_seqForwardPass_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getSimpleCLQT()

        steps = 10

        x0 = np.array([1,2,3])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        xs1a, us1a = clqt_tf.clqt_seq_forwardpass(x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=False)
        xs1b, us1b = clqt_tf.clqt_seq_forwardpass(x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=True)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=False)
        u_list2b, x_list2b = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=True)

        Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
        ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
        Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
        vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        self.assertTrue(err < 1e-5)

        us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
        xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)
        us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
        xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

        t = tf.range(0, steps, dtype=tf.float64) * dt
        plt.plot(t, xs1a[1:])
        plt.plot(t, xs2a[1:], '--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(us1a - us2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1a - xs2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1b - us2b))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1b - xs2b))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_seqForwardPass_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        x0 = np.array([1,2])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        xs1a, us1a = clqt_tf.clqt_seq_forwardpass(x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=False)
        xs1b, us1b = clqt_tf.clqt_seq_forwardpass(x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=True)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)
        u_list2a, x_list2a = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=False)
        u_list2b, x_list2b = clqt.seqForwardPass(x0, Kx_list2, d_list2, u_zoh=True)

        us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
        xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)
        us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
        xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

        t = tf.range(0, steps, dtype=tf.float64) * dt
        plt.plot(t, xs1a[1:])
        plt.plot(t, xs2a[1:], '--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(us1a - us2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1a - xs2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1b - us2b))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1b - xs2b))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_seqForwardPass_3(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        t0_list = []
        ST_list = []
        vT_list = []
        x0_list = []
        for i in range(5):
            dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
            t0 = 0.1 * i
            ST = ST + 0.01 * i * tf.eye(ST.shape[0], dtype=ST.dtype)
            vT = vT + 0.01 * i
            x0 = np.array([1,2]) + 0.1 * i
            t0_list.append(t0)
            ST_list.append(ST)
            vT_list.append(vT)
            x0_list.append(x0)

        t0 = tf.convert_to_tensor(t0_list, dtype=tf.float64)
        ST = tf.convert_to_tensor(ST_list, dtype=tf.float64)
        vT = tf.convert_to_tensor(vT_list, dtype=tf.float64)
        x0 = tf.convert_to_tensor(x0_list, dtype=tf.float64)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        xs1a, us1a = clqt_tf.clqt_seq_forwardpass(x0, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=False)
        xs1b, us1b = clqt_tf.clqt_seq_forwardpass(x0, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=True)

        for i in range(len(t0_list)):
            Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps, t0=t0_list[i],
                                                                       S=ST_list[i].numpy(),
                                                                       v=vT_list[i].numpy())

            u_list2a, x_list2a = clqt.seqForwardPass(x0_list[i], Kx_list2, d_list2, t0=t0_list[i], u_zoh=False)
            u_list2b, x_list2b = clqt.seqForwardPass(x0_list[i], Kx_list2, d_list2, t0=t0_list[i], u_zoh=True)

            us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
            xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)
            us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
            xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

            err = tf.reduce_max(tf.math.abs(us1a[i, ...] - us2a))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(xs1a[i, ...] - xs2a))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(us1b[i, ...] - us2b))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(xs1b[i, ...] - xs2b))
            print(err)
            self.assertTrue(err < 1e-5)

    def test_seqFwdBwdPass_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getSimpleCLQT()

        steps = 10

        x0 = np.array([1,2,3])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0_tf, steps, T)

        As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps)

        As2 = tf.convert_to_tensor(A_list2, dtype=tf.float64)
        bs2 = tf.convert_to_tensor(b_list2, dtype=tf.float64)
        Cs2 = tf.convert_to_tensor(C_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        self.assertTrue(err < 1e-5)

    def test_seqFwdBwdPass_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        x0 = np.array([1,2])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0_tf, steps, T)

        As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps)

        As2 = tf.convert_to_tensor(A_list2, dtype=tf.float64)
        bs2 = tf.convert_to_tensor(b_list2, dtype=tf.float64)
        Cs2 = tf.convert_to_tensor(C_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        self.assertTrue(err < 1e-5)

    def test_seqFwdBwdPass_3(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        t0_list = []
        A0_list = []
        b0_list = []
        C0_list = []
        x0_list = []
        for i in range(5):
            x0 = np.array([1, 2]) + 0.1 * i
            x0_tf = tf.constant(x0, dtype=tf.float64)
            dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0_tf, steps, T)
            t0 = 0.1 * i
            A0 = A0 + 0.01 * i * tf.eye(A0.shape[0], dtype=A0.dtype)
            b0 = b0 + 0.01 * i
            C0 = C0 + 0.01 * i * tf.eye(C0.shape[0], dtype=C0.dtype)
            t0_list.append(t0)
            A0_list.append(A0)
            b0_list.append(b0)
            C0_list.append(C0)
            x0_list.append(x0)

        t0 = tf.convert_to_tensor(t0_list, dtype=tf.float64)
        A0 = tf.convert_to_tensor(A0_list, dtype=tf.float64)
        b0 = tf.convert_to_tensor(b0_list, dtype=tf.float64)
        C0 = tf.convert_to_tensor(C0_list, dtype=tf.float64)
        x0 = tf.convert_to_tensor(x0_list, dtype=tf.float64)

        As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        for i in range(len(t0_list)):
            A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps, t0=t0_list[i],
                                                           A=A0_list[i].numpy(),
                                                           b=b0_list[i].numpy(),
                                                           C=C0_list[i].numpy())

            As2 = tf.convert_to_tensor(A_list2, dtype=tf.float64)
            bs2 = tf.convert_to_tensor(b_list2, dtype=tf.float64)
            Cs2 = tf.convert_to_tensor(C_list2, dtype=tf.float64)

            err = tf.reduce_max(tf.math.abs(As1[i, ...] - As2))
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(bs1[i, ...] - bs2))
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(Cs1[i, ...] - Cs2))
            self.assertTrue(err < 1e-5)

    def test_combineFwdBwd_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        x0 = np.array([1,2])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0_tf, steps, T)
        As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        xs1, us1 = clqt_tf.clqt_combine_fwdbwd(Kxs1, ds1, Ss1, vs1, As1, bs1, Cs1)

        A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps)
        Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps)
        u_list2, x_list2 = clqt.combineForwardBackward(Kx_list2, d_list2, S_list2, v_list2, A_list2, b_list2, C_list2)

        us2 = tf.convert_to_tensor(u_list2, dtype=tf.float64)
        xs2 = tf.convert_to_tensor(x_list2, dtype=tf.float64)

        t = tf.range(0, steps, dtype=tf.float64) * dt
        plt.plot(t, xs1[1:])
        plt.plot(t, xs2[1:], '--')
        plt.show()

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_combineFwdBwd_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        steps = 100

        t0_list = []
        A0_list = []
        b0_list = []
        C0_list = []
        ST_list = []
        vT_list = []
        x0_list = []
        for i in range(5):
            x0 = np.array([1, 2]) + 0.1 * i
            x0_tf = tf.constant(x0, dtype=tf.float64)
            dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T)
            dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0_tf, steps, T)
            t0 = 0.1 * i
            A0 = A0 + 0.01 * i * tf.eye(A0.shape[0], dtype=A0.dtype)
            b0 = b0 + 0.01 * i
            C0 = C0 + 0.01 * i * tf.eye(C0.shape[0], dtype=C0.dtype)
            ST = ST + 0.01 * i * tf.eye(ST.shape[0], dtype=ST.dtype)
            vT = vT + 0.01 * i
            t0_list.append(t0)
            A0_list.append(A0)
            b0_list.append(b0)
            C0_list.append(C0)
            ST_list.append(ST)
            vT_list.append(vT)
            x0_list.append(x0)

        t0 = tf.convert_to_tensor(t0_list, dtype=tf.float64)
        A0 = tf.convert_to_tensor(A0_list, dtype=tf.float64)
        b0 = tf.convert_to_tensor(b0_list, dtype=tf.float64)
        C0 = tf.convert_to_tensor(C0_list, dtype=tf.float64)
        ST = tf.convert_to_tensor(ST_list, dtype=tf.float64)
        vT = tf.convert_to_tensor(vT_list, dtype=tf.float64)
        x0 = tf.convert_to_tensor(x0_list, dtype=tf.float64)

        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        xs1, us1 = clqt_tf.clqt_combine_fwdbwd(Kxs1, ds1, Ss1, vs1, As1, bs1, Cs1)

        for i in range(len(t0_list)):
            Kx_list2, d_list2, S_list2, v_list2 = clqt.seqBackwardPass(steps, t0=t0_list[i],
                                                                       S=ST_list[i].numpy(),
                                                                       v=vT_list[i].numpy())
            A_list2, b_list2, C_list2 = clqt.seqFwdBwdPass(x0, steps, t0=t0_list[i],
                                                           A=A0_list[i].numpy(),
                                                           b=b0_list[i].numpy(),
                                                           C=C0_list[i].numpy())
            u_list2, x_list2 = clqt.combineForwardBackward(Kx_list2, d_list2, S_list2, v_list2,
                                                           A_list2, b_list2, C_list2)

            us2 = tf.convert_to_tensor(u_list2, dtype=tf.float64)
            xs2 = tf.convert_to_tensor(x_list2, dtype=tf.float64)

            err = tf.reduce_max(tf.math.abs(us1[i, ...] - us2))
            print(err)
            self.assertTrue(err < 1e-5)

            err = tf.reduce_max(tf.math.abs(xs1[i, ...] - xs2))
            print(err)
            self.assertTrue(err < 1e-5)

    def test_parBackwardPass_init_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        As1, bs1, Cs1, etas1, Js1 = clqt_tf.par_backwardpass_init_bw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        elems = clqt.parBackwardPass_init(blocks, steps, forward=False)

        As2 = tf.convert_to_tensor([elem[0] for elem in elems], dtype=tf.float64)
        bs2 = tf.convert_to_tensor([elem[1] for elem in elems], dtype=tf.float64)
        Cs2 = tf.convert_to_tensor([elem[2] for elem in elems], dtype=tf.float64)
        etas2 = tf.convert_to_tensor([elem[3] for elem in elems], dtype=tf.float64)
        Js2 = tf.convert_to_tensor([elem[4] for elem in elems], dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(etas1 - etas2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Js1 - Js2))
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_init_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        As1, bs1, Cs1, etas1, Js1 = clqt_tf.par_backwardpass_init_fw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        elems = clqt.parBackwardPass_init(blocks, steps, forward=True)

        As2 = tf.convert_to_tensor([elem[0] for elem in elems], dtype=tf.float64)
        bs2 = tf.convert_to_tensor([elem[1] for elem in elems], dtype=tf.float64)
        Cs2 = tf.convert_to_tensor([elem[2] for elem in elems], dtype=tf.float64)
        etas2 = tf.convert_to_tensor([elem[3] for elem in elems], dtype=tf.float64)
        Js2 = tf.convert_to_tensor([elem[4] for elem in elems], dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(etas1 - etas2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Js1 - Js2))
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)

        Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
        ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
        Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
        vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        self.assertTrue(err < 1e-5)

    def test_parBackwardPass_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps, forward=True)

        Kxs2 = tf.convert_to_tensor(Kx_list2, dtype=tf.float64)
        ds2 = tf.convert_to_tensor(d_list2, dtype=tf.float64)
        Ss2 = tf.convert_to_tensor(S_list2, dtype=tf.float64)
        vs2 = tf.convert_to_tensor(v_list2, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        self.assertTrue(err < 1e-5)

    def test_parForwardPass_init_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)
        Psis1, phis1 = clqt_tf.clqt_par_forwardpass_init_fw(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        elems2 = clqt.parForwardPass_init(x0, Kx_list2, d_list2, blocks, steps, forward=True)

        Psis2 = tf.convert_to_tensor([elem[0] for elem in elems2], dtype=tf.float64)
        phis2 = tf.convert_to_tensor([elem[1] for elem in elems2], dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Psis1 - Psis2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(phis1 - phis2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_parForwardPass_init_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)
        Psis1, phis1 = clqt_tf.clqt_par_forwardpass_init_bw(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        elems2 = clqt.parForwardPass_init(x0, Kx_list2, d_list2, blocks, steps, forward=False)

        Psis2 = tf.convert_to_tensor([elem[0] for elem in elems2], dtype=tf.float64)
        phis2 = tf.convert_to_tensor([elem[1] for elem in elems2], dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(Psis1 - Psis2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(phis1 - phis2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_parForwardPass_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)
        xs1a, us1a = clqt_tf.clqt_par_forwardpass(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, forward=True, u_zoh=False)
        xs1b, us1b = clqt_tf.clqt_par_forwardpass(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, forward=True, u_zoh=True)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        u_list2a, x_list2a = clqt.parForwardPass(x0, Kx_list2, d_list2, blocks, steps, forward=True, u_zoh=False)
        u_list2b, x_list2b = clqt.parForwardPass(x0, Kx_list2, d_list2, blocks, steps, forward=True, u_zoh=True)

        us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
        xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)
        us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
        xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(us1a - us2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1a - xs2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1b - us2b))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1b - xs2b))
        print(err)
        self.assertTrue(err < 1e-5)


    def test_parForwardPass_2(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)
        xs1a, us1a = clqt_tf.clqt_par_forwardpass(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, forward=False, u_zoh=False)
        xs1b, us1b = clqt_tf.clqt_par_forwardpass(blocks, steps, x0_tf, Kxs1, ds1, dt, t0, F_f, L_f, c_f, forward=False, u_zoh=True)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        u_list2a, x_list2a = clqt.parForwardPass(x0, Kx_list2, d_list2, blocks, steps, forward=False, u_zoh=False)
        u_list2b, x_list2b = clqt.parForwardPass(x0, Kx_list2, d_list2, blocks, steps, forward=False, u_zoh=True)

        us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
        xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)
        us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
        xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(us1a - us2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1a - xs2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1b - us2b))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1b - xs2b))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_parFwdPwdPass_init_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0 = clqt_tf.clqt_par_fwdbwdpass_defaults(blocks, steps, T)
        As1, bs1, Cs1, etas1, Js1 = clqt_tf.par_fwdbwdpass_init(blocks, steps, x0_tf, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True)

        elems = clqt.parFwdBwdPass_init(x0, blocks, steps, forward=True)

        As2 = tf.convert_to_tensor([elem[0] for elem in elems], dtype=tf.float64)
        bs2 = tf.convert_to_tensor([elem[1] for elem in elems], dtype=tf.float64)
        Cs2 = tf.convert_to_tensor([elem[2] for elem in elems], dtype=tf.float64)
        etas2 = tf.convert_to_tensor([elem[3] for elem in elems], dtype=tf.float64)
        Js2 = tf.convert_to_tensor([elem[4] for elem in elems], dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(As1 - As2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(bs1 - bs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Cs1 - Cs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(etas1 - etas2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Js1 - Js2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_parFwdPwdPass_1(self):
        clqt, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = self.getRandomCLQT()

        blocks = 5
        steps = 20

        x0 = np.array([2,1])
        x0_tf = tf.constant(x0, dtype=tf.float64)

        dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
        Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)
#        dt, t0 = clqt_tf.clqt_par_fwdbwdpass_defaults(blocks, steps, T)
        xs1a, us1a = clqt_tf.clqt_par_fwdbwdpass(blocks, steps, x0_tf, Ss1, vs1, Kxs1, ds1, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True)
        xs1b, us1b = clqt_tf.clqt_par_fwdbwdpass(blocks, steps, x0_tf, Ss1, vs1, Kxs1, ds1, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False)

        Kx_list2, d_list2, S_list2, v_list2 = clqt.parBackwardPass(blocks, steps)
        u_list2a, x_list2a = clqt.parFwdBwdPass(x0, Kx_list2, d_list2, S_list2, v_list2, blocks, steps, forward=True)
        u_list2b, x_list2b = clqt.parFwdBwdPass(x0, Kx_list2, d_list2, S_list2, v_list2, blocks, steps, forward=False)

        us2a = tf.convert_to_tensor(u_list2a, dtype=tf.float64)
        xs2a = tf.convert_to_tensor(x_list2a, dtype=tf.float64)

        us2b = tf.convert_to_tensor(u_list2b, dtype=tf.float64)
        xs2b = tf.convert_to_tensor(x_list2b, dtype=tf.float64)

        err = tf.reduce_max(tf.math.abs(us1a - us2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1a - xs2a))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1b - us2b))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1b - xs2b))
        print(err)
        self.assertTrue(err < 1e-5)
