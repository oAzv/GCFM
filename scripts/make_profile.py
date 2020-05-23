#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import multiprocessing
import os
import re
import subprocess

import numpy as np
import pandas as pd
from Bio import SeqIO

from scripts.utils import get_annotated_struct, align_anno_seq


def run_DAFS(path):
    inn = subprocess.Popen(['../scripts-non-py/dafs', path], stdout=subprocess.PIPE, encoding='utf-8')
    buf = []
    while True:
        innline = inn.stdout.readline()
        buf.append(innline)

        if not innline:
            break

    return buf[4].strip().upper(), buf[6].strip().upper()


def run_RNAfold(path):
    inn = subprocess.Popen(['../scripts-non-py/RNAfold', '-p', '--noPS', path], stdout=subprocess.PIPE,
                           encoding='utf-8')
    buf = []

    while True:
        innline = inn.stdout.readline()
        # print(len(innline))
        buf.append(innline)
        # print(buf2)
        if not innline:
            break

    return re.split('[\s]', buf[2].strip().upper())[0], re.split('[\s]', buf[8].strip().upper())[0]


def get_matrix(input, n):
    mat = np.zeros((n, n), dtype=np.float32)
    image_mat = np.zeros((n, n), dtype=np.float32)
    tag1 = '%start of base pair probability data\n'
    tag2 = 'showpage\n'
    tag3 = 'ubox'
    tag4 = 'lbox'

    switch_tag = 0

    with open(input) as f:
        for line in f:
            while (switch_tag == 1) & (tag3 in line):
                try:
                    ele = re.split('[\s]', line.rstrip(' ubox\n'))
                    mat[int(ele[0]) - 1, int(ele[1]) - 1] = float(ele[2]) ** 2
                    image_mat[int(ele[0]) - 1, int(ele[1]) - 1] = (float(ele[2]) ** 2) * 255
                except:
                    pass
                break
            while (switch_tag == 1) & (tag4 in line):
                try:
                    ele_image = re.split('[\s]', line.rstrip(' lbox\n'))
                    image_mat[int(ele_image[1]) - 1, int(ele_image[0]) - 1] = (float(ele_image[2]) ** 2) * 255
                except:
                    pass
                break
            if line == tag1:
                switch_tag = 1
            elif line == tag2:
                break
            else:
                continue
    return mat, image_mat


def cal_bp_pro(mat):
    bp = np.zeros((2, len(mat[0]), 3), dtype=np.float32)
    for i in range(2):
        # each base
        for j in range(len(mat[i])):
            L = 0
            R = 0
            # probability of (
            for k in range(j, len(mat[i])):
                L += mat[i][j][k]
            # probability of )
            for k in range(j):
                R += mat[i][k][j]
            bp[i][j][0] = L
            bp[i][j][1] = R
            bp[i][j][2] = 1 - L - R
    return bp


def make_8bit(pair, ss):
    j = 0
    for i in range(len(ss)):
        if ss[i] == '.':
            while pair[i + j] == '-':
                j += 1
            pair = pair[:i + j] + pair[i + j].lower() + pair[i + j + 1:]
        else:
            while pair[i + j] == '-':
                j += 1
    return pair


def get_image_fea(features_dir, tag, itr1, j):
    fx_path = os.path.join(features_dir, tag + '.pkl')

    wk = pd.read_pickle(fx_path)

    dl1 = wk[wk['name'] == str(dataset[itr1][0])]['vector'].values[0]
    dl2 = wk[wk['name'] == str(dataset[j][0])]['vector'].values[0]
    return dl1, dl2


def make_pairFASTA(dataset, itr1, outpath):
    train_data = np.empty((0, 800, 38), dtype=np.float32)
    train_label = np.empty((0), dtype=np.int32)
    for j in range(itr1 + 1, len(dataset)):

        path_to_pairFasta = "./pair" + str(itr1) + "," + str(j) + ".fa"
        pairFa = ""
        for k in [dataset[itr1], dataset[j]]:
            pairFa += ">" + k[0] + "\n" + k[2] + "\n"
        with open(path_to_pairFasta, 'w') as f:
            f.write(pairFa)

        pair1, pair2 = run_DAFS(path_to_pairFasta)
        ss1, ss2 = run_RNAfold(path_to_pairFasta)

        pss_path = '../tools-non-py/lib/parse_secondary_structure'
        anos1 = align_anno_seq(get_annotated_struct(ss1, pss_path), pair1)
        anos2 = align_anno_seq(get_annotated_struct(ss2, pss_path), pair2)

        pair1 = make_8bit(pair1, ss1)
        pair2 = make_8bit(pair2, ss2)

        subprocess.call(["rm", path_to_pairFasta])

        dp1 = './' + str(dataset[itr1][0]) + '_dp.ps'
        dp2 = './' + str(dataset[j][0]) + '_dp.ps'

        mat1, mat2 = get_matrix(dp1, len(pair1))

        b_mat = [mat1]
        mat1, mat2 = get_matrix(dp2, len(pair2))
        b_mat = b_mat + [mat1]
        bp_mat = cal_bp_pro(b_mat)

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

        if dataset[itr1][1] == dataset[j][1]:
            label = 1
        else:
            label = 0

        features_dir = '../features/image_features/'
        tags = ['alexnet', 'resnet101', 'vgg16']

        r1 = data.shape[1] // 2
        for tag in tags:
            dl1, dl2 = get_image_fea(features_dir, tag, itr1, j)

            data = np.insert(data, r1, values=dl1, axis=1)
            data = np.insert(data, r1 * 2 + 1, values=dl2, axis=1)
            r1 += 1

        train_data = np.append(train_data, np.array([data]), axis=0)
        train_label = np.append(train_label, label)

    outdata = outpath + "/portion/ncRNApair_data" + str(itr1) + ".npy"
    outlabel = outpath + "/portion/ncRNApair_label" + str(itr1) + ".npy"

    np.save(outdata, train_data)
    np.save(outlabel, train_label)

    print('%d completed' % itr1)


if __name__ == '__main__':
    start = datetime.datetime.now()

    infile = '../data/sequence_all.fa'
    itr1 = 0
    outpath = '../features/profile/'

    outpath = outpath.rstrip('/')

    dataset = []
    out = ""
    for record in SeqIO.parse(infile, "fasta"):
        itr1 += 1

        id_part = record.id
        id_parts = id_part.split(",")
        seq_part = str(record.seq.upper())

        dataset = dataset + [[id_parts[0], int(id_parts[1]), seq_part]]

        out += id_parts[0] + ":" + id_parts[1] + "\n"

    with open(outpath + "/genelabel.txt", 'w') as f:
        f.write(out)

    debug = False

    # single_debug
    if debug == True:
        make_pairFASTA(dataset, 0, outpath)
    else:
        jobs = []
        for i in range(itr1):
            p = multiprocessing.Process(target=make_pairFASTA, args=(dataset, i, outpath))
            jobs.append(p)
            p.start()

        for job in jobs:
            while job.is_alive() == True:
                continue
            else:
                pass

    os.system('rm *.ps')
    end = datetime.datetime.now()
    print(end - start)
