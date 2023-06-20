"""
Unit tests for Viterbi with TF.

@author: Simo Särkkä
"""

import tensorflow as tf
import parallel_control.fsc_tf as fsc_tf
import parallel_control.ge_model_np as ge_model_np
import parallel_control.viterbi_np as viterbi_np
import parallel_control.viterbi_tf as viterbi_tf

import unittest

mm = tf.linalg.matmul
mv = tf.linalg.matvec
top = tf.linalg.matrix_transpose


class Viterbi_tf_UnitTest(unittest.TestCase):
    """Unit tests for Viterbi_tf"""

    def test_fsc_seq_viterbi(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.seqViterbi(y_list)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        v_map2, Vs_vit2 = viterbi_tf.viterbi_fsc_seq_bwfw(fs, Ls, LT)

#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs_vit2.dtype) - Vs_vit2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_map1, dtype=v_map2.dtype) - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_fsc_par_bwfw_viterbi(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.parBwdFwdViterbi(y_list)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        v_map2, Vs_vit2 = viterbi_tf.viterbi_fsc_par_bwfw(fs, Ls, LT)

#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs_vit2.dtype) - Vs_vit2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_map1, dtype=v_map2.dtype) - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_fsc_par_bwfwbw_viterbi(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.parBwdFwdBwdViterbi(y_list)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        v_map2, Vs_vit2 = viterbi_tf.viterbi_fsc_par_bwfwbw(fs, Ls, LT)

#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs_vit2.dtype) - Vs_vit2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_map1, dtype=v_map2.dtype) - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_fsc_seq_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_fsc_seq_bwfw_speedtest(model, steps)

    def test_fsc_par_bwfw_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_fsc_par_bwfw_speedtest(model, steps)

    def test_fsc_par_bwfwbw_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_fsc_par_bwfwbw_speedtest(model, steps)

    def test_seq_forwardpass(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)

        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)
        us, Vs = viterbi_tf.viterbi_seq_forwardpass(tf_prior, tf_Pi, tf_Po, ys)

        print(V_list1)
        print(Vs)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs.dtype) - Vs))
        print(err)
        self.assertTrue(err < 1e-10)


    def test_seq_backwardpass(self):
        ge = ge_model_np.GEModel(seed=5)
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us, Vs = viterbi_tf.viterbi_seq_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        v_map2 = viterbi_tf.viterbi_seq_backwardpass(us, Vs[-1, :])

        print(V_list1[-1, :])
        print(Vs[-1, :])

        print(v_map1)
        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_map1, dtype=v_map2.dtype) - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_seq_fwbw(self):
        ge = ge_model_np.GEModel(seed=5)
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        v_map2, Vs = viterbi_tf.viterbi_seq_fwbw(tf_prior, tf_Pi, tf_Po, ys)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(v_map1, dtype=v_map2.dtype) - v_map2))
        print(err)
        self.assertTrue(err <= 2)

        err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(V_list1, dtype=Vs.dtype) - Vs))
        print(err)
        self.assertTrue(err < 1e-10)

    def test_viterbi_par_forwardpass_init(self):
        ge = ge_model_np.GEModel(seed=5)
        x_list, y_list = ge.genData(10)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        Vs = viterbi_tf.viterbi_par_forwardpass_init(tf_prior, tf_Pi, tf_Po, ys)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        elems_most = fsc_tf.fsc_par_backwardpass_init_most(fs, Ls)
        elems_last = fsc_tf.fsc_par_backwardpass_init_last(LT)

        elems = tf.concat([elems_most, tf.expand_dims(elems_last, 0)], axis=0)
        elems = top(tf.reverse(elems, axis=[0]))

        print(Vs)
        print(elems)

        err = tf.math.reduce_max(tf.math.abs(elems - Vs))
        print(err)
        self.assertTrue(err < 1e-10)


    def test_viterbi_par_forwardpass(self):
        ge = ge_model_np.GEModel(seed=5)
        x_list, y_list = ge.genData(10)

#        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
#        us1, Vs1 = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
#        us1 = tf.reverse(us1, axis=[0])
#        Vs1 = tf.reverse(Vs1, axis=[0])

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us1, Vs1 = viterbi_tf.viterbi_seq_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        print(us1)
        print(us2)

        err = tf.math.reduce_max(tf.math.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(us1 - us2))
        print(err)
        self.assertTrue(err == 0)

    def test_viterbi_par_backwardpass_init(self):
        ge = ge_model_np.GEModel(seed=5)
        x_list, y_list = ge.genData(10)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        us1, Vs1 = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
        x0 = tf.argmin(Vs1[0], output_type=tf.int32)

        first_elem = fsc_tf.fsc_par_forwardpass_init_first(x0, fs.shape[1])
        most_elems = fsc_tf.fsc_par_forwardpass_init_most(fs, us1)

        elems1 = tf.concat([tf.expand_dims(first_elem, 0), most_elems], axis=0)
        elems1 = tf.reverse(elems1, axis=[0])

        print(elems1)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
#        us1 = tf.reverse(us1, axis=[0])
#        Vs1 = tf.reverse(Vs1, axis=[0])
        elems2 = viterbi_tf.viterbi_par_backwardpass_init(us2, Vs2[-1])

        print(elems2)

        err = tf.math.reduce_max(tf.math.abs(elems1 - elems2))
        print(err)
        self.assertTrue(err == 0)

    def test_viterbi_par_backwardpass(self):
        ge = ge_model_np.GEModel(seed=5)

        x_list, y_list = ge.genData(200)

#        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)
#        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us1, Vs1 = viterbi_tf.viterbi_seq_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        v_map1 = viterbi_tf.viterbi_seq_backwardpass(us1, Vs1[-1, :])

        us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        v_map2 = viterbi_tf.viterbi_par_backwardpass(us2, Vs2[-1, :])

#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
        print(err)
        self.assertTrue(err == 0)

        def test_viterbi_par_backwardpass(self):
            ge = ge_model_np.GEModel(seed=5)

            x_list, y_list = ge.genData(200)

            #        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)
            #        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

            ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
            tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
            tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
            tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

            us1, Vs1 = viterbi_tf.viterbi_seq_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
            v_map1 = viterbi_tf.viterbi_seq_backwardpass(us1, Vs1[-1, :])

            us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
            v_map2 = viterbi_tf.viterbi_par_backwardpass(us2, Vs2[-1, :])

            #        print(v_map1)
            #        print(v_map2)

            err = tf.math.reduce_max(tf.math.abs(Vs1 - Vs2))
            print(err)
            self.assertTrue(err < 1e-10)

            err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
            print(err)
            self.assertTrue(err == 0)

    def test_viterbi_par_fwbw(self):
        ge = ge_model_np.GEModel(seed=5)

        x_list, y_list = ge.genData(200)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        v_map1, Vs1 = viterbi_tf.viterbi_seq_fwbw(tf_prior, tf_Pi, tf_Po, ys)
        v_map2, Vs2 = viterbi_tf.viterbi_par_fwbw(tf_prior, tf_Pi, tf_Po, ys)

        err = tf.math.reduce_max(tf.math.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
        print(err)
        self.assertTrue(err <= 2)

    def test_viterbi_seqpar_fwbw(self):
        ge = ge_model_np.GEModel(seed=5)

        x_list, y_list = ge.genData(200)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int64)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        v_map1, Vs1 = viterbi_tf.viterbi_seq_fwbw(tf_prior, tf_Pi, tf_Po, ys)
        v_map2, Vs2 = viterbi_tf.viterbi_seqpar_fwbw(tf_prior, tf_Pi, tf_Po, ys)

        err = tf.math.reduce_max(tf.math.abs(Vs1 - Vs2))
        print(err)
        self.assertTrue(err < 1e-10)

        err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_viterbi_par_fwdbwdfwdpass_init(self):
        ge = ge_model_np.GEModel(seed=5)
        x_list, y_list = ge.genData(10)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        us1, Vs1 = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
        x0 = tf.argmin(Vs1[0], output_type=tf.int32)

        first_elem = fsc_tf.fsc_par_fwdbwdpass_init_first(x0, Ls[0, ...])
        most_elems = fsc_tf.fsc_par_fwdbwdpass_init_most(fs, Ls)

        elems1 = tf.concat([tf.expand_dims(first_elem, 0), most_elems], axis=0)
        elems1 = tf.reverse(top(elems1), axis=[0])

#        print(x0)
#        print(elems1)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        elems2 = viterbi_tf.viterbi_par_bwfwpass_init(tf_prior, tf_Pi, tf_Po, ys)
#        print(xT)
#        print(elems2)

        err = tf.math.reduce_max(tf.math.abs(elems1[:-1] - elems2[:-1]))  # Last one will differ
        print(err)
        self.assertTrue(err < 1e-10)

    def test_viterbi_par_fwdbwdfwdpass(self):
        ge = ge_model_np.GEModel(seed=5)
        x_list, y_list = ge.genData(50)

        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)
        v_map0, V_list0 = viterbi.parBwdFwdBwdViterbi(y_list)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        v_map1, Vs_vit1 = viterbi_tf.viterbi_fsc_par_bwfwbw(fs, Ls, LT)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        us2, Vs2 = viterbi_tf.viterbi_par_forwardpass(tf_prior, tf_Pi, tf_Po, ys)
        v_map2, Vfs2 = viterbi_tf.viterbi_par_bwfwpass(tf_prior, tf_Pi, tf_Po, ys, Vs2)

#        print(v_map0)
#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(v_map0 - v_map2))
        print(err)
        self.assertTrue(err == 0)

        err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
        print(err)
        self.assertTrue(err == 0)


    def test_viterbi_par_fwbwfw(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map0, V_list0 = viterbi.parBwdFwdBwdViterbi(y_list)

        fs, Ls, LT = viterbi_tf.make_viterbi_fsc(ge, y_list)
        v_map1, Vs1 = viterbi_tf.viterbi_fsc_par_bwfwbw(fs, Ls, LT)

        Vs0 = tf.convert_to_tensor(V_list0, dtype=Vs1.dtype)

        ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
        tf_prior = tf.convert_to_tensor(ge.prior, dtype=tf.float64)
        tf_Pi = tf.convert_to_tensor(ge.Pi, dtype=tf.float64)
        tf_Po = tf.convert_to_tensor(ge.Po, dtype=tf.float64)

        v_map2, Vs2 = viterbi_tf.viterbi_par_fwbwfw(tf_prior, tf_Pi, tf_Po, ys)

#        print(v_map0)
#        print(v_map1)
#        print(v_map2)

        err = tf.math.reduce_max(tf.math.abs(v_map1 - v_map2))
        print(err)
        self.assertTrue(err == 0)

    def test_seq_fwbw_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_seq_fwbw_speedtest(model, steps)

    def test_par_fwbw_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_par_fwbw_speedtest(model, steps)

    def test_par_fwbwfw_viterbi_speedtest(self):
        model = ge_model_np.GEModel()
        for steps in [100, 200, 300, 400, 500]:
            elapsed, err = viterbi_tf.viterbi_par_fwbwfw_speedtest(model, steps)

