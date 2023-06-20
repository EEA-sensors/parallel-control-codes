#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run continuous-time linear tracking experiments-hjb.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import parallel_control.contlin_speedtest as clinspeed

import sys, os
import pprint
from pathlib import Path

def write_to_file(T,D,run_times,errors,id,config):
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
        for s,n,t,e in zip(T,D,run_times,errors):
            file.write("%d %d %f %f\n" % (s,n,t,e))

    cfilename = 'res/%s_%s_%s_config.txt' % (config["model"],id,devtxt)
    print("Saving config to '%s'" % cfilename)

    with open(cfilename, "w") as cfile:
        pprint.PrettyPrinter(indent=4, stream=cfile).pprint(config)


def main(argv):
    config = {
        "model" : "tracking",
        "start" : 2,
        "stop" : 3,
        "num" : 5,
        "steps" : 10,
        "n" : 4,
        "niter" : 2,
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
        elif tmp[0] == 'niter':
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

    # XXX: This might help in segmentation faults sometimes, but right now not
    enable_growth = False

    if enable_growth:
        physical_devices = tf.config.list_physical_devices('CPU')
        try:
            for dev in physical_devices:
                print(f"Enabling memory growth of {dev}")
                tf.config.experimental.set_memory_growth(dev, True)
        except ValueError:
            print(f'Invalid device (CPU).')
            pass
        except RuntimeError:
            print(f'Cannot modify virtual device (CPU) once initialized.')
            pass

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for dev in physical_devices:
                print(f"Enabling memory growth of {dev}")
                tf.config.experimental.set_memory_growth(dev, True)
        except ValueError:
            print(f'Invalid device (GPU).')
            pass
        except RuntimeError:
            print(f'Cannot modify virtual device (GPU) once initialized.')
            pass

    if config["single"] != 0:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    dtype = tf.float64 # For now this is what we use
    #n_iter = 100       # This is fixed for now
    n_iter = 10       # This is fixed for now

    start = config["start"]
    stop = config["stop"]
    num = config["num"]
    steps = config["steps"]
    niter = config["niter"]
    B_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('B_list = %s' % str(B_list))

    if config["model"] == "tracking":
        model = clinspeed.clqr_get_tracking_model(dtype=dtype)
        D = 4
    elif config["model"] == "mass":
        print("Not implemented model '%s'" % config["model"])
        sys.exit(1)
    else:
        print("Unknown model '%s'" % config["model"])
        sys.exit(1)

    if (config["mask"] & 1) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_seq_bw_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_seq_bw' % config["n"], config)

    if (config["mask"] & 2) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_seq_bw_fw_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_seq_bw_fw' % config["n"], config)

    if (config["mask"] & 4) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_par_bw_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_par_bw' % config["n"], config)

    if (config["mask"] & 8) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_par_bw_fw_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_par_bw_fw' % config["n"], config)

    if (config["mask"] & 16) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_par_bw_fwbw_speedtest(model, int(blocks), int(steps), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_par_bw_fwbw' % config["n"], config)

    if (config["mask"] & 32) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_parareal_bw_speedtest(model, int(blocks), int(steps), int(niter), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_parareal_%d_bw' % (config["n"], niter), config)

    if (config["mask"] & 64) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_parareal_bw_fw_speedtest(model, int(blocks), int(steps), int(niter), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_parareal_%d_bw_fw' % (config["n"], niter), config)

    if (config["mask"] & 128) != 0:
        run_times = []
        err_list = []
        dims = []
        for blocks in B_list:
            elapsed, err = clinspeed.clqt_parareal_bw_fwbw_speedtest(model, int(blocks), int(steps), int(niter), n_iter=n_iter, device=config["device"])
            run_times.append(elapsed)
            err_list.append(err)
            dims.append(D)
        write_to_file(B_list, dims, run_times, err_list, '%d_parareal_%d_bw_fwbw' % (config["n"], niter), config)


if __name__ == "__main__":
   main(sys.argv)
