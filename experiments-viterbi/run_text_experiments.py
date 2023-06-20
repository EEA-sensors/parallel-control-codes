"""
Run Viterbi text experiments.

@author: Simo Särkkä
"""

import time
import tensorflow as tf
import parallel_control.viterbi_tf as viterbi_tf
import numpy as np
import parallel_control.text_model as tm

import sys, os
import pprint
from pathlib import Path

def write_to_file(name,methods,run_times,errors,config):
    devtxt = ''
    for ch in config["device"]:
        if ch.isalpha():
            devtxt += ch.lower()

    if config["single"] != 0:
        devtxt += "_sc"
    else:
        devtxt += "_mc"

    order = config["order"]
    filename = 'res/%s_%d_%s.txt' % (name,order,devtxt)
    print("Saving data to '%s'" % filename)

    with open(filename, "w") as file:
        for m,t,e in zip(methods,run_times,errors):
            file.write("%d %d %f %f\n" % (m,order,t,e))

    cfilename = 'res/%s_%d_%s_config.txt' % (name,order,devtxt)
    print("Saving config to '%s'" % cfilename)

    with open(cfilename, "w") as cfile:
        pprint.PrettyPrinter(indent=4, stream=cfile).pprint(config)


def text_error(text1, text2):
    err = 0.0
    for i in range(len(text1)):
        if text1[i] != text2[i]:
            err += 1
    return err / len(text1)


def main(argv):
    config = {
        "order" : 1,
        "single" : 0,
        "device" : '/CPU:0',
        "mask" : 65535
    }

    usage = 'Usage: %s' % Path(argv[0]).name
    for key, val in config.items():
        usage += ' [%s=%s]' % (key, str(val))

    for i, arg in enumerate(sys.argv[1:]):
        tmp = arg.split('=')
        if len(tmp) != 2:
            print(usage)
            sys.exit(1)
        if tmp[0] == 'order':
            config[tmp[0]] = int(tmp[1])
        elif tmp[0] == 'single':
            config[tmp[0]] = int(tmp[1])
        elif tmp[0] == 'device':
            config[tmp[0]] = tmp[1]
        elif tmp[0] == 'mask':
            config[tmp[0]] = int(tmp[1])
        else:
            print(usage)
            sys.exit(1)

    print("-" * 40)
    pprint.PrettyPrinter(indent=4).pprint(config)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("-" * 40)

    if config["single"] != 0:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    dtype = tf.float64
    itype = tf.int32

    n_iter = 10       # This is fixed for now

    with open('../notebooks/text1.txt') as f:
        text1 = f.read()

    with open('../notebooks/text2.txt') as f:
        text2 = f.read()

    order = config["order"]
    device = config["device"]

    text_model = tm.TextModel(errp=0.1, order=order)
    text3 = text_model.makeNoisyText(text2)

    Po = text_model.getNthOrderPo()
    prior = text_model.getNthOrderPrior(text1)
    Pi = text_model.getNthOrderPi(text1)

    xs = text_model.textToIndex(text2)
    ys = text_model.textToIndex(text3)

    tf_ys = tf.convert_to_tensor(ys, dtype=itype)
    tf_prior = tf.convert_to_tensor(prior, dtype=dtype)
    tf_Pi = tf.convert_to_tensor(Pi, dtype=dtype)
    tf_Po = tf.convert_to_tensor(Po, dtype=dtype)


    raw_err = text_error(text2, text3)
    print(f"{raw_err = }")

    methods = []
    run_times = []
    err_list = []

    print(f"{device = }")

    if (config["mask"] & 1) != 0:
        print("Running seq_fwbw...")
        with tf.device(device):
            zs_tf, Vs_tf = viterbi_tf.viterbi_seq_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            tic = time.time()
            for i in range(n_iter):
                zs_tf, Vs_tf = viterbi_tf.viterbi_seq_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            toc = time.time()
            elapsed = (toc - tic) / n_iter

        text4 = text_model.indexToText(zs_tf[1:])
        err = text_error(text2, text4)
        print(f"{err = }")
        print('Took %f s (err = %f).' % (elapsed, err))

        methods.append(1)
        run_times.append(elapsed)
        err_list.append(err)

    if (config["mask"] & 2) != 0:
        print("Running seqpar_fwbw...")
        with tf.device(device):
            zs_tf, Vs_tf = viterbi_tf.viterbi_seqpar_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            tic = time.time()
            for i in range(n_iter):
                zs_tf, Vs_tf = viterbi_tf.viterbi_seqpar_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            toc = time.time()
            elapsed = (toc - tic) / n_iter

        text4 = text_model.indexToText(zs_tf[1:])
        err = text_error(text2, text4)
        print(f"{err = }")
        print('Took %f s (err = %f).' % (elapsed, err))

        methods.append(2)
        run_times.append(elapsed)
        err_list.append(err)

    if (config["mask"] & 4) != 0:
        print("Running par_fwbw...")
        with tf.device(device):
            zs_tf, Vs_tf = viterbi_tf.viterbi_par_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            tic = time.time()
            for i in range(n_iter):
                zs_tf, Vs_tf = viterbi_tf.viterbi_par_fwbw(tf_prior, tf_Pi, tf_Po, tf_ys)
            toc = time.time()
            elapsed = (toc - tic) / n_iter

        text4 = text_model.indexToText(zs_tf[1:])
        err = text_error(text2, text4)
        print(f"{err = }")
        print('Took %f s (err = %f).' % (elapsed, err))

        methods.append(3)
        run_times.append(elapsed)
        err_list.append(err)

    if (config["mask"] & 8) != 0:
        print("Running par_fwbwfw...")
        with tf.device(device):
            zs_tf, Vs_tf = viterbi_tf.viterbi_par_fwbwfw(tf_prior, tf_Pi, tf_Po, tf_ys)
            tic = time.time()
            for i in range(n_iter):
                zs_tf, Vs_tf = viterbi_tf.viterbi_par_fwbwfw(tf_prior, tf_Pi, tf_Po, tf_ys)
            toc = time.time()
            elapsed = (toc - tic) / n_iter

        text4 = text_model.indexToText(zs_tf[1:])
        err = text_error(text2, text4)
        print(f"{err = }")
        print('Took %f s (err = %f).' % (elapsed, err))

        methods.append(4)
        run_times.append(elapsed)
        err_list.append(err)

    write_to_file('text', methods, run_times, err_list, config)

if __name__ == "__main__":
   main(sys.argv)
