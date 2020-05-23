#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import pickle
import time
from multiprocessing.pool import Pool

import chainer
import gc as gc_m
import numpy as np
import pandas as pd

from scripts.utils import check_path


def run_cnn(X_train, X_test, y_train, y_test, i, c_size, cnn_model_path, retrain=False):
    cnn_train = chainer.datasets.TupleDataset(X_train, y_train)
    cnn_test = chainer.datasets.TupleDataset(X_test, y_test)
    from scripts.CNN import CNNmodel as Model

    batchsize = 128  # 128
    epoch = 10
    gpu = -1
    out = './logs/cnn/result'

    output_ch1 = 64  # number of 1st convolutional layer's filters
    output_ch2 = 128  # number of 2nd convolutional layer's filters
    filter_height = 15  # filter size of convolutional layer
    n_units = 1632  # 1632 # number of middel layer's units
    n_label = 2
    width = X_train.shape[3]
    outdir = out + str(i)

    RC = Model.Runchainer(batchsize, outdir, output_ch1, output_ch2,
                          filter_height, width, n_units, n_label)
    if retrain == True:
        model = RC.learn(cnn_train, cnn_test, gpu, epoch, cnn_model_path)
        acc, f1 = RC.predict(X_test, y_test, cnn_model_path)
        print([acc, f1])
        return acc, f1, RC
    else:
        acc, f1 = RC.predict(X_test, y_test, cnn_model_path)
        print([acc, f1])
        return acc, f1, RC


def rc_extract(data):
    data = RC.extract(data[np.newaxis, :, :, :], '../model/cnn_5' + '.model')

    return data.array[0]


if __name__ == '__main__':

    dir = '../features/profile'
    print("Running...")
    data = dir + "/ncRNApair_data" + ".npy"
    label = dir + "/ncRNApair_label" + ".npy"

    X = np.load(data)

    # select feature
    sp = 19
    spm = X.shape[2] // 2
    X = X[..., np.r_[: sp, spm:spm + sp]]

    y = np.load(label)
    from sklearn.utils import indexable
    from sklearn.model_selection import StratifiedKFold

    result_cnn = []
    b_size = len(y)
    c_size = b_size
    X, y = indexable(X, y)
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)

    cnn_time = 0
    i = 0
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        test = False
        if test:
            b_size = 2000
            c_size = 2000
            nt = int(b_size * 0.9) - 1
            ot = int(b_size * 0.1) - 1

            X_train = X_train[0:nt]
            y_train = y_train[0:nt]
            X_test = X_test[0:ot]
            y_test = y_test[0:ot]

        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

        start_cnn = time.time()
        cnn_model_path = '../model/cnn_' + str(i) + '.model'
        retrain = True
        acc, f1, RC = run_cnn(X_train, X_test, y_train, y_test, i, c_size, retrain=retrain,
                              cnn_model_path=cnn_model_path)

        re_data_path = '../features/profile/re_data/'
        check_path(re_data_path)
        train_path = re_data_path + 'Xc_train_' + str(i) + '.pkl'
        test_path = re_data_path + 'Xc_test_' + str(i) + '.pkl'
        ytr_path = re_data_path + 'y_train_' + str(i) + '.pkl'
        yte_path = re_data_path + 'y_test_' + str(i) + '.pkl'

        extract_t1 = time.time()

        pool1 = Pool(os.cpu_count() - 1)
        Xc_train = np.array(pool1.map(rc_extract, X_train))
        pool1.close()
        pool1.join()

        pool2 = Pool(os.cpu_count() - 1)
        Xc_test = np.array(pool2.map(rc_extract, X_test))
        pool2.close()
        pool2.join()

        with open(train_path, "wb") as f:
            pickle.dump(Xc_train, f, pickle.HIGHEST_PROTOCOL)
        with open(test_path, "wb") as f:
            pickle.dump(Xc_test, f, pickle.HIGHEST_PROTOCOL)

        with open(ytr_path, "wb") as f:
            pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)
        with open(yte_path, "wb") as f:
            pickle.dump(y_test, f, pickle.HIGHEST_PROTOCOL)

        result_cnn.append([acc, f1])
        cnn_time += (time.time() - start_cnn)

        del X_train, X_test, y_train, y_test
        gc_m.collect()
        i += 1
        break

    from scripts.utils import check_path
    from scripts.gcforest.utils.log_utils import strftime

    tem_path = '../output/cnn_model_train/'
    check_path(tem_path)

    result_cnn = pd.DataFrame(result_cnn, columns=['acc', 'f1'])
    result_cnn.loc['mean'] = result_cnn.apply(lambda x: x.mean())

    result_cnn.to_csv(tem_path + 'cnn_model_' + str(b_size) + '_' + str(cnn_time) + '_result_' + strftime() + '.csv')
