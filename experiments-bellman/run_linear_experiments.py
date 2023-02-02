#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run linear tracking experiments-bellman.

@author: Simo Särkkä
"""

import tensorflow as tf
import parallel_control.linear_speedtest as linspeed
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

    filename = 'res/%s_%s_%s.txt' % (config["model"],id,devtxt)
    print("Saving to '%s'" % filename)

    with open(filename, "w") as file:
        for s,n,t in zip(T,D,run_times):
            file.write("%d %d %f\n" % (s,n,t))

def get_generator(config, dtype):
    if config["model"] == "tracking":
        lqt_gen = linspeed.tracking_generator(start=config["start"], stop=config["stop"], num=config["num"], dtype=dtype)
    elif config["model"] == "mass":
        lqt_gen = linspeed.mass_generator(N=config["n"], start=config["start"], stop=config["stop"], num=config["num"], dtype=dtype)
    else:
        print("Unknown model '%s'" % config["model"])
        sys.exit(1)
    return lqt_gen

def main(argv):
    config = {
        "model" : "tracking",
        "start" : 2,
        "stop" : 5,
        "num" : 20,
        "n" : 4,
        "nc" : 4,
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
        elif tmp[0] == 'n':
            config[tmp[0]] = int(tmp[1])
        elif tmp[0] == 'nc':
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
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_sequential_bw_speedtest(lqt_gen, device=config["device"])
        write_to_file(T, D, run_times, '%d_sequential_bw' % config["n"], config)

    if (config["mask"] & 2) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_parallel_bw_speedtest(lqt_gen, device=config["device"])
        write_to_file(T, D, run_times, '%d_parallel_bw' % config["n"], config)

    if (config["mask"] & 4) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_sequential_bwfw_speedtest(lqt_gen, device=config["device"])
        write_to_file(T, D, run_times, '%d_sequential_bwfw' % config["n"], config)

    if (config["mask"] & 8) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_parallel_bwfw_1_speedtest(lqt_gen, device=config["device"])
        write_to_file(T, D, run_times, '%d_parallel_bwfw_1' % config["n"], config)

    if (config["mask"] & 16) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_parallel_bwfw_2_speedtest(lqt_gen, device=config["device"])
        write_to_file(T, D, run_times, '%d_parallel_bwfw_2' % config["n"], config)

    if (config["mask"] & 32) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_sequential_cond_bwfw_speedtest(lqt_gen, device=config["device"], Nc=config["nc"])
        write_to_file(T, D, run_times, '%d_sequential_cond%d_bwfw' % (config["n"], config["nc"]), config)

    if (config["mask"] & 64) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_parallel_cond_bwfw_1_speedtest(lqt_gen, device=config["device"], Nc=config["nc"])
        write_to_file(T, D, run_times, '%d_parallel_cond%d_bwfw_1' % (config["n"], config["nc"]), config)

    if (config["mask"] & 128) != 0:
        lqt_gen = get_generator(config, dtype)
        T, D, run_times = linspeed.lqt_parallel_cond_bwfw_2_speedtest(lqt_gen, device=config["device"], Nc=config["nc"])
        write_to_file(T, D, run_times, '%d_parallel_cond%d_bwfw_2' % (config["n"], config["nc"]), config)


if __name__ == "__main__":
   main(sys.argv)

