"""
Unit tests for continuous-time linear speedtests.

@author: Simo Särkkä
"""

import unittest
import tensorflow as tf
import parallel_control.linear_model_tf as linear_model_tf
import parallel_control.contlin_speedtest as clinspeed
import math

class ContLinSpeedtest_UnitTest(unittest.TestCase):
    def test_ref_1(self):
        model = clinspeed.clqr_get_tracking_model()
        blocks = 100
        steps = 100
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_ref_sol_2(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-2)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-1)

    def test_clqt_seq_bw(self):
        blocks = 50
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2 = clinspeed.clqt_seq_bw(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_clqt_seq_bw_speedtest(self):
        blocks = 100
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        for i in [1,2,3]:
            elapsed, err = clinspeed.clqt_seq_bw_speedtest(model, i*blocks, steps)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_seq_bw_fw(self):
        blocks = 50
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_seq_bw_fw(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_clqt_seq_bw_fw_speedtest(self):
        blocks = 100
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        for i in [1,2,3]:
            elapsed, err = clinspeed.clqt_seq_bw_fw_speedtest(model, i*blocks, steps)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_par_bw(self):
        blocks = 50
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2 = clinspeed.clqt_par_bw(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_clqt_par_bw_speedtest(self):
        blocks = 100
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        for i in [1,2,3]:
            elapsed, err = clinspeed.clqt_par_bw_speedtest(model, i*blocks, steps)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_par_bw_fw(self):
        blocks = 50
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_par_bw_fw(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_clqt_par_bw_fw_speedtest(self):
        blocks = 100
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        for i in [1,2,3]:
            elapsed, err = clinspeed.clqt_par_bw_fw_speedtest(model, i*blocks, steps)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_par_bw_fwbw(self):
        blocks = 50
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_2(blocks, steps, *model)
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_par_bw_fwbw(blocks, steps, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-5)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-5)

    def test_clqt_par_bw_fwbw_speedtest(self):
        blocks = 100
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        for i in [1,2,3]:
            elapsed, err = clinspeed.clqt_par_bw_fwbw_speedtest(model, i*blocks, steps)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_parareal_bw(self):
        blocks = 300
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        parareal_niter = 2
        Ss2, vs2, Kxs2, ds2 = clinspeed.clqt_parareal_bw(blocks, steps, parareal_niter, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-4)

    def test_clqt_parareal_bw_speedtest(self):
        blocks = 100
        steps = 50

        parareal_niter = 1
        model = clinspeed.clqr_get_tracking_model()
        for i in [3,4,5]:
            elapsed, err = clinspeed.clqt_parareal_bw_speedtest(model, i*blocks, steps, parareal_niter=parareal_niter)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_parareal_bw_fw(self):
        blocks = 300
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_1(blocks, steps, *model)
        parareal_niter = 2
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_parareal_bw_fw(blocks, steps, parareal_niter, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-3)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-2)

    def test_clqt_parareal_bw_fw_speedtest(self):
        blocks = 100
        steps = 50

        parareal_niter = 1
        model = clinspeed.clqr_get_tracking_model()
        for i in [3,4,5]:
            elapsed, err = clinspeed.clqt_parareal_bw_fw_speedtest(model, i*blocks, steps, parareal_niter=parareal_niter)
            print(f"elapsed = {elapsed}, err = {err}")

    def test_clqt_parareal_bw_fwbw(self):
        blocks = 300
        steps = 50

        model = clinspeed.clqr_get_tracking_model()
        Ss1, vs1, Kxs1, ds1, xs1, us1 = clinspeed.clqt_ref_sol_2(blocks, steps, *model)
        parareal_niter = 2
        Ss2, vs2, Kxs2, ds2, xs2, us2 = clinspeed.clqt_parareal_bw_fwbw(blocks, steps, parareal_niter, *model)

        err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(vs1 - vs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(Kxs1 - Kxs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(ds1 - ds2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(xs1 - xs2))
        print(err)
        self.assertTrue(err < 1e-4)

        err = tf.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err < 1e-4)

    def test_clqt_parareal_bw_fwbw_speedtest(self):
        blocks = 100
        steps = 50

        parareal_niter = 1
        model = clinspeed.clqr_get_tracking_model()
        for i in [3,4,5]:
            elapsed, err = clinspeed.clqt_parareal_bw_fwbw_speedtest(model, i*blocks, steps, parareal_niter=parareal_niter)
            print(f"elapsed = {elapsed}, err = {err}")

