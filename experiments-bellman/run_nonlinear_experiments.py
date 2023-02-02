#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run nonlinear experiments-bellman.

@author: Simo Särkkä
"""

import tensorflow as tf
import parallel_control.nonlinear_speedtest as nlinspeed
import sys, os
import pprint
from pathlib import Path

def write_to_file(T,D,run_times,id,config):
    devtxt = ''
    for ch in config["device"]:
        if ch.isalpha():
            devtxt += ch.lower()

    if config["single"] != 0:
        devtxt += "_sc"
    else:
        devtxt += "_mc"

    filename = 'res/nonlinear_%s_%s.txt' % (id,devtxt)
    print("Saving to '%s'" % filename)

    with open(filename, "w") as file:
        for s,n,t in zip(T,D,run_times):
            file.write("%d %d %f\n" % (s,n,t))

def main(argv):
    config = {
        "start" : 2,
        "stop" : 5,
        "num" : 20,
        "single" : 0,
        "device" : '/CPU:0',
        "mask" : 7
    }

    usage = 'Usage: %s' % Path(argv[0]).name
    for key, val in config.items():
        usage += ' [%s=%s]' % (key, str(val))

    for i, arg in enumerate(sys.argv[1:]):
        tmp = arg.split('=')
        if len(tmp) != 2:
            print(usage)
            sys.exit(1)
        elif tmp[0] == 'start':
            config[tmp[0]] = float(tmp[1])
        elif tmp[0] == 'stop':
            config[tmp[0]] = float(tmp[1])
        elif tmp[0] == 'num':
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

    if (config["mask"] & 1) != 0:
        nlqt_gen = nlinspeed.nonlinear_generator(start=config["start"], stop=config["stop"], num=config["num"], dtype=dtype)
        T, D, run_times  = nlinspeed.nlqt_iter_seq_speedtest(nlqt_gen, device=config["device"])
        write_to_file(T, D, run_times, 'seq', config)

    if (config["mask"] & 2) != 0:
        nlqt_gen = nlinspeed.nonlinear_generator(start=config["start"], stop=config["stop"], num=config["num"], dtype=dtype)
        T, D, run_times  = nlinspeed.nlqt_iter_par_1_speedtest(nlqt_gen, device=config["device"])
        write_to_file(T, D, run_times, 'par_1', config)

    if (config["mask"] & 4) != 0:
        nlqt_gen = nlinspeed.nonlinear_generator(start=config["start"], stop=config["stop"], num=config["num"], dtype=dtype)
        T, D, run_times  = nlinspeed.nlqt_iter_par_2_speedtest(nlqt_gen, device=config["device"])
        write_to_file(T, D, run_times, 'par_2', config)

if __name__ == "__main__":
   main(sys.argv)
