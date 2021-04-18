#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import pickle
import subprocess
import warnings

import numpy as np
import pandas as pd

from scripts.utils import check_path, run_DAFS, run_RNAfold, make_8bit, \
    get_matrix, cal_bp_pro, save_image, wk_image, get_annotated_struct, align_anno_seq

warnings.filterwarnings('ignore')

__doc__ = """
prediction - a script that predicts a pairwise ncRNA
====================================================

**prediction** is a Python script for pairwise ncRNA prediction.

Main Functions
--------------
Here are just a few of the things that **prediction** does well:

  - Complete pairwise ncRNA relation prediction.

Main Program Functions
----------------------

"""

def get_seq_pair(data, path_to_pairFasta):
    '''Build all the ncRNA data sets.

    Args:

        data: A pairwise sequence of string formats.

        path_to_pairFasta: Files that temporarily store pairs of sequences.

    Returns:

        ncRNA data sets.
    '''
    import re
    path_to_pairFasta.write(data)
    path_to_pairFasta.seek(0)
    dataset = []

    data = re.split('\n', data)
    for i in range(int(len(data) / 2)):
        dataset = dataset + [[data[i * 2].lstrip('>'), data[i * 2 + 1]]]
    return dataset


def get_image_fea(dataset, features_dir, tag):
    '''Obtain the image features obtained by the pre-training model.

    Args:

        dataset: ncRNA data sets.

        features_dir: The path of the image feature.

        tag: The name of the pre-training model.

    Returns:
        Image features.
    '''
    fx_path = os.path.join(features_dir, tag + '.pkl')
    wk = pd.read_pickle(fx_path)
    dl1 = wk[wk['name'] == str(dataset[0][0])]['vector'].values[0]
    dl2 = wk[wk['name'] == str(dataset[1][0])]['vector'].values[0]
    return dl1, dl2


def make_pairFASTA(dataset, env, fa_path):
    '''Construct multi-view feature representation.

    Args:

        dataset: All the ncRNA data.

        env: The environment in which the predicted results are stored.

        fa_path: Files that temporarily store pairs of sequences.

    Returns:

        multi-view feature representation.

    '''
    train_data = np.empty((0, 800, 38), dtype=np.float32)

    pair1, pair2 = run_DAFS(fa_path)
    ss1, ss2, buf = run_RNAfold(fa_path)
    pair1 = make_8bit(pair1, ss1)
    pair2 = make_8bit(pair2, ss2)

    pss_path = '../tools-non-py/lib/parse_secondary_structure'
    anos1 = align_anno_seq(get_annotated_struct(ss1, pss_path), pair1)
    anos2 = align_anno_seq(get_annotated_struct(ss2, pss_path), pair2)

    dp1 = './' + str(dataset[0][0]) + '_dp.ps'
    dp2 = './' + str(dataset[1][0]) + '_dp.ps'

    mat1, mat2 = get_matrix(dp1, len(pair1))
    b_mat = [mat1]
    image_mat = [mat2]

    mat1, mat2 = get_matrix(dp2, len(pair2))
    b_mat = b_mat + [mat1]
    image_mat = image_mat + [mat2]

    image_path = env + '/image/'
    import shutil
    shutil.rmtree(image_path)
    check_path(image_path)

    re_ = save_image(dataset, image_mat, image_path)

    del image_mat
    while re_ == 'Done':
        subprocess.call(["rm", dp1])
        subprocess.call(["rm", dp2])
        break

    re_ = wk_image(env, image_path)
    while re_ == 'Done':
        break

    bp_mat = cal_bp_pro(b_mat)
    del b_mat

    data = np.zeros(([800, 32]), dtype=np.float32)
    n1 = 0
    n2 = 0
    for k in range(len(pair1)):

        if pair1[k] == "A":
            data[k][0] = 1
        elif pair1[k] == "C":
            data[k][1] = 1
        elif pair1[k] == "G":
            data[k][2] = 1
        elif pair1[k] == "T":
            data[k][3] = 1
        elif pair1[k] == "a":
            data[k][4] = 1
        elif pair1[k] == "c":
            data[k][5] = 1
        elif pair1[k] == "g":
            data[k][6] = 1
        elif pair1[k] == "t":
            data[k][7] = 1
        elif pair1[k] == "-":
            data[k][8] = 1

        if anos1[k] == "B":
            data[k][9] = 1
        elif anos1[k] == "H":
            data[k][10] = 1
        elif anos1[k] == "M":
            data[k][11] = 1
        elif anos1[k] == "T":
            data[k][12] = 1
        if pair1[k] != "-":
            data[k][13] = bp_mat[0][n1][0]
            data[k][14] = bp_mat[0][n1][1]
            data[k][15] = bp_mat[0][n1][2]
            n1 += 1

        if pair2[k] == "A":
            data[k][16] = 1
        elif pair2[k] == "C":
            data[k][17] = 1
        elif pair2[k] == "G":
            data[k][18] = 1
        elif pair2[k] == "T":
            data[k][19] = 1
        elif pair2[k] == "a":
            data[k][20] = 1
        elif pair2[k] == "c":
            data[k][21] = 1
        elif pair2[k] == "g":
            data[k][22] = 1
        elif pair2[k] == "t":
            data[k][23] = 1
        elif pair2[k] == "-":
            data[k][24] = 1

        if anos2[k] == "B":
            data[k][25] = 1
        elif anos2[k] == "H":
            data[k][26] = 1
        elif anos2[k] == "M":
            data[k][27] = 1
        elif anos2[k] == "T":
            data[k][28] = 1
        if pair2[k] != "-":
            data[k][29] = bp_mat[1][n2][0]
            data[k][30] = bp_mat[1][n2][1]
            data[k][31] = bp_mat[1][n2][2]
            n2 += 1
    features_dir = env
    tags = ['alexnet', 'resnet101', 'vgg16']

    r1 = data.shape[1] // 2

    for tag in tags:
        dl1, dl2 = get_image_fea(dataset, features_dir, tag)
        data = np.insert(data, r1, values=dl1, axis=1)
        data = np.insert(data, r1 * 2 + 1, values=dl2, axis=1)
        r1 += 1

    train_data = np.append(train_data, np.array([data]), axis=0)

    outdata = env + "ncRNApair_data.npy"

    np.save(outdata, train_data)
    return train_data


def rc_extract(data, model='../model/cnn_4' + '.model'):
    '''Use CNN module to extract the features of the data.

        Args:

            data: Data for features to be extracted.

            model: CNN Model store and read paths.

        Returns:

            The data processed by CNN.
    '''
    from scripts.CNN import CNNmodel as Model
    batchsize = 128  # 128
    epoch = 10
    gpu = -1
    out = './logs/cnn/result'

    output_ch1 = 64
    output_ch2 = 128
    filter_height = 15
    n_units = 1632
    n_label = 2

    width = data.shape[2]
    outdir = out

    RC = Model.Runchainer(batchsize, outdir, output_ch1, output_ch2,
                          filter_height, width, n_units, n_label)

    data = RC.extract(data[np.newaxis, :, :, :], model)

    return data.array[0]


def gcfm_predict(seq1, seq2):
    '''Predicting the results of pairwise sequences.

    Args:

        seq1: ncRNA nucleotide sequence A.

        seq2: ncRNA nucleotide sequence B.

    Returns:
        Predicted results.
    '''
    prediction_env = '../output/prediction'
    result = prediction_env + '/results/'
    from tempfile import NamedTemporaryFile as NTF
    path_to_pairFasta = NTF(mode='w+t', suffix='.fa', dir=prediction_env + '/tmp', encoding='utf-8', delete=True)

    check_path(result)
    data = seq1 + '\n' + seq2

    dataset = get_seq_pair(data, path_to_pairFasta)
    data = make_pairFASTA(dataset, prediction_env, path_to_pairFasta.name)

    model = '../model/cnn_4' + '.model'
    data = rc_extract(data, model)
    data = data[np.newaxis, :]
    gc_model_path = '../model/gc_4' + '.pkl'
    with open(gc_model_path, 'rb') as f:
        gc = pickle.load(f)
    y_pred = gc.predict(data)

    return y_pred[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GCFM classification:')
    parser.add_argument('--seq1', '-s1',
                        default='>SNORD16_ENST00000362803.1\nTGCAATGATGTCGTAATTTGCGTCTTACTCTGTTCTCAGCGACAGTTGCCTGCTGTCAGTAAGCTGGTACAGAAGGTTGACGAAAATTCTTACTGAGCA',
                        help='seq1')
    parser.add_argument('--seq2', '-s2',
                        default='>SNORA58_ENST00000505219.1\nGGGCATACTCGTAGACCTTGCCTGACTGTGCTCATGTCCAGGCAGGGGGGACAGTGTATGCAAGAATAATTTGGAGTTCCTGCCAGCTCTAACCAGCTTCATCAGTGGCTGGATAAATTGCAGGACTCTAAACATTT',
                        help='seq2')
    args = parser.parse_args()

    gcfm_predict(args.seq1, args.seq2)
