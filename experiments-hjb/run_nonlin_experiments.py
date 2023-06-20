#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run continuous-time nonlinear HJB experiments.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import parallel_control.hjb_grid_1d_speedtest as hjbspeed

import sys, os
import pprint
from pathlib import Path

def write_to_file(T,D,run_times,errors1,errors2,id,config):
    devtxt = ''
    for ch in config["device"]:
        if ch.isalpha():
            devtxt += ch.lower()

    if config["single"] != 0:
        devtxt += "_sc"
    else:
        devtxt += "_mc"

    filename = 'res/%s_%s_%s.txt' % (config["model"],id,devtxt)
    print("Saving data to '%s'" % filename)

    with open(filename, "w") as file:
        for s,n,t,e1,e2 in zip(T,D,run_times,errors1,errors2):
            file.write("%d %d %f %f %f\n" % (s,n,t,e1,e2))

    cfilename = 'res/%s_%s_%s_config.txt' % (config["model"],id,devtxt)
    print("Saving config to '%s'" % cfilename)

    with open(cfilename, "w") as cfile:
        pprint.PrettyPrinter(indent=4, stream=cfile).pprint(config)

def main(argv):
    config = {
        "model" : "velocity",
        "start" : 2,
        "stop" : 3,
        "num" : 5,
        "steps" : 10,
        "n" : 20,
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
        if tmp[0] == 'model':
            config[tmp[0]] = tmp[1]
        elif tmp[0] == 'start':
            config[tmp[0]] = float(tmp[1])
        elif tmp[0] == 'stop':
            config[tmp[0]] = float(tmp[1])
        elif tmp[0] == 'num':
            config[tmp[0]] = int(tmp[1])
        elif tmp[0] == 'steps':
            config[tmp[0]] = int(tmp[1])
        elif tmp[0] == 'n':
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

    dtype = tf.float64 # For now this is what we use

    n_iter = 10       # This is fixed for now

    start = config["start"]
    stop = config["stop"]
    num = config["num"]
    steps = config["steps"]
    n = config["n"]
    D = n
    B_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('B_list = %s' % str(B_list))

    model = hjbspeed.get_model(n, n)

    if (config["mask"] & 1) != 0:
        run_times = []
        err_list1 = []
        err_list2 = []
        dims = []
        for blocks in B_list:
            elapsed, err1, err2 = hjbspeed.seq_upwind_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list1.append(err1)
            err_list2.append(err2)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list1, err_list2, '%d_seq_upwind' % n, config)

    if (config["mask"] & 2) != 0:
        run_times = []
        err_list1 = []
        err_list2 = []
        dims = []
        for blocks in B_list:
            elapsed, err1, err2 = hjbspeed.seq_assoc_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list1.append(err1)
            err_list2.append(err2)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list1, err_list2, '%d_seq_assoc' % n, config)

    if (config["mask"] & 4) != 0:
        run_times = []
        err_list1 = []
        err_list2 = []
        dims = []
        for blocks in B_list:
            elapsed, err1, err2 = hjbspeed.par_assoc_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list1.append(err1)
            err_list2.append(err2)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list1, err_list2, '%d_par_assoc' % n, config)

if __name__ == "__main__":
   main(sys.argv)
