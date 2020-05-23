#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import multiprocessing
import time

import gc
import numpy as np


def assembledata_sp10(ddir, n=10, *, mulit=True, stop_num=1):
    checkdata = np.load(ddir + "/portion/ncRNApair_data0.npy")
    dsize = len(checkdata) + 1
    genelen = len(checkdata[0])
    width = len(checkdata[0][0])

    def process_data10(n, j):
        data = np.empty((0, genelen, width), dtype=np.float32)
        label = np.empty((0), dtype=np.int32)
        for i in range(dsize):
            indata = ddir + "/portion/ncRNApair_data" + str(i) + ".npy"
            inlabel = ddir + "/portion/ncRNApair_label" + str(i) + ".npy"

            tmpdata = np.load(indata)
            tmplabel = np.load(inlabel)

            tmpdata_10 = np.array_split(tmpdata, n, axis=0)
            tmplabel_10 = np.array_split(tmplabel, n, axis=0)

            data = np.concatenate([data, tmpdata_10[j]], axis=0)
            label = np.concatenate([label, tmplabel_10[j]], axis=0)
            del tmpdata_10, tmplabel_10
            gc.collect()

        np.save(ddir + "/data_10/ncRNApair_data" + str(j) + ".npy", data)
        np.save(ddir + "/data_10/ncRNApair_label" + str(j) + ".npy", label)
        del data, label
        gc.collect()

    if mulit:
        jobs = []
        for j in range(stop_num):
            p = multiprocessing.Process(target=process_data10, args=(n, j))
            jobs.append(p)
            p.start()

        for job in jobs:
            while job.is_alive() == True:
                continue
            else:
                pass

    else:
        for j in range(stop_num):
            process_data10(j)

    return


def assembledata(ddir):
    checkdata = np.load(ddir + "/portion/ncRNApair_data0.npy")
    dsize = len(checkdata) + 1
    genelen = len(checkdata[0])
    width = len(checkdata[0][0])

    data = np.empty((0, genelen, width), dtype=np.float32)
    label = np.empty((0), dtype=np.int32)

    print("Makeing data...")
    start = time.time()

    for i in range(dsize):
        indata = ddir + "/portion/ncRNApair_data" + str(i) + ".npy"
        inlabel = ddir + "/portion/ncRNApair_label" + str(i) + ".npy"

        tmpdata = np.load(indata)
        tmplabel = np.load(inlabel)

        data = np.concatenate([data, tmpdata], axis=0)
        label = np.concatenate([label, tmplabel], axis=0)

    elapsed_time = time.time() - start
    print("Loding time:{0}".format(elapsed_time) + "[sec]")
    print("Data size :", len(data), "*", len(data[0]), "*", len(data[0][0]))

    np.save(ddir + "/ncRNApair_data.npy", data)
    np.save(ddir + "/ncRNApair_label.npy", label)


if __name__ == '__main__':
    datadir = '../features/profile/'

    print("Makeing data...")
    start = time.time()

    assembledata(datadir)

    print("Complete makeing data.")
    spend_time = time.time() - start
    print("Loding time:{0}".format(spend_time) + "[sec]")
