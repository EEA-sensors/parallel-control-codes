"""
Unit tests for Viterbi with numpy.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.ge_model_np as ge_model_np
import parallel_control.viterbi_np as viterbi_np

import unittest


class Viterbi_np_UnitTest(unittest.TestCase):
    """Unit tests for Viterbi_np"""

    def setupRefData(self):
        # This is from Matlab reference
        x_list = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,
                           1,3,3,3,3,2,3,3,3,2,2,2,2,2,2,2,2,3,3,3,
                           3,3,3,3,3,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,
                           0,0,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0])
        y_list = np.array([1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,0,
                           1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,
                           1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,
                           0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        b_list = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
                           0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                           1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
                           0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0])
        m_list = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                           2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                           2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,
                           0,0,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        return x_list, y_list, b_list, m_list


    def test_ge_model1(self):
        x_list, y_list, b_list, m_list = self.setupRefData()

        b_list2 = ge_model_np.get_b_list(x_list)

        self.assertTrue(max(abs(b_list - b_list2)) == 0)

    def test_ref_viterbi(self):
        x_list, y_list, b_list, m_list = self.setupRefData()

        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        v_map, V_list = viterbi.ref_viterbi(y_list)
        print(v_map[1:])
        print(m_list)

        self.assertTrue(max(abs(v_map[1:] - m_list)) == 0)

    def test_ref_viterbi_log(self):

        ge = ge_model_np.GEModel(seed=10)
        x_list, y_list = ge.genData(20)
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)
        print(x_list)
        print(y_list)

        v_map1, V_list1 = viterbi.ref_viterbi(y_list)
        v_map2, V_list2 = viterbi.ref_viterbi_log(y_list)
        print(v_map1)
        print(v_map2)
        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

    def test_ref_viterbi_log_v2(self):

        ge = ge_model_np.GEModel(seed=10)
        x_list, y_list = ge.genData(30)
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)
        print(x_list)
        print(y_list)

        v_map1, V_list1 = viterbi.ref_viterbi_log(y_list)
        v_map2, V_list2 = viterbi.ref_viterbi_log_v2(y_list)
        print(v_map1)
        print(v_map2)
        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)


    def test_fsc_1(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(200)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        fsc = viterbi.getFSC(y_list)
        u_list2, V_list2 = fsc.seqBackwardPass()

        V_list2.reverse()
        V_list2b = np.array(V_list2[:])

        err = linalg.norm(V_list1 - V_list2b)
        self.assertTrue(err < 1e-10)


    def test_fsc_2(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(50)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        fsc = viterbi.getFSC(y_list)
        u_list2, V_list2 = fsc.seqBackwardPass()
        x0 = np.argmin(V_list2[0])
        min_u_list, min_x_list = fsc.seqForwardPass(x0, u_list2)
        min_u_list.reverse()
        min_x_list.reverse()
        v_map2 = np.array(min_x_list[:])

        print(v_map1)
        print(v_map2)
        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

    def test_fsc_3(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)

        fsc = viterbi.getFSC(y_list)
        u_list2, V_list2 = fsc.seqBackwardPass()
        x0 = np.argmin(V_list2[0])
        min_u_list, min_x_list = fsc.seqForwardPass(x0, u_list2)
        v_map2, V_list2 = viterbi.fscToViterbi(min_x_list, V_list2)

        print(v_map1)
        print(v_map2)
        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

        print(V_list1)
        print(V_list2)

        err = max([linalg.norm(e1 - e2) for e1,e2 in zip(V_list1, V_list2)])
        self.assertTrue(err < 1e-10)

    def test_seq_viterbi(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)
        v_map2, V_list2 = viterbi.seqViterbi(y_list)

        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

        err = linalg.norm(V_list1 - V_list2)
        self.assertTrue(err < 1e-10)


    def test_par_viterbi_1(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)
        v_map2, V_list2 = viterbi.parBwdFwdViterbi(y_list)

        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

        err = linalg.norm(V_list1 - V_list2)
        self.assertTrue(err < 1e-10)


    def test_par_viterbi_2(self):
        ge = ge_model_np.GEModel()
        viterbi = viterbi_np.Viterbi_np(ge.prior, ge.Pi, ge.Po)

        x_list, y_list = ge.genData(100)

        v_map1, V_list1 = viterbi.ref_viterbi_log_v2(y_list)
        v_map2, V_list2 = viterbi.parBwdFwdBwdViterbi(y_list)

        print(x_list)
        print(v_map1)
        print(v_map2)

        self.assertTrue(max(abs(v_map1 - v_map2)) == 0)

        err = linalg.norm(V_list1 - V_list2)
        self.assertTrue(err < 1e-10)
