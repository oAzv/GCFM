#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import datetime
import multiprocessing

import pandas as pd
from Bio import SeqIO

from scripts.utils import *

__doc__ = """
make_profile - a scripts for multi-view feature representation construction
===========================================================================

**make_profile** is a Python script that provides calling utility functions to construct multi-view feature representations.

Main Functions
--------------
Here are just a few of the things that **make_profile** does well:

  - Read gray image features.
  - One-hot encodes nucleotide secondary structure fusion sequence and structural subunit sequence.
  - Combination of multi-view feature representations.

Main Program Functions
----------------------

"""

def get_image_fea(features_dir:str, tag:str, i:int, j:int):
    '''Get the image features extracted from the pre-training model.

    Args:

        features_dir: Image feature path.

        tag: The name of the pre-trained image model.

        i: The ith a ncRNA.

        j: The jth a ncRNA.

    Returns:

        list(array like): Image features.
    '''
    fx_path = os.path.join(features_dir, tag + '.pkl')

    wk = pd.read_pickle(fx_path)

    dl1 = wk[wk['name'] == str(dataset[i][0])]['vector'].values[0]
    dl2 = wk[wk['name'] == str(dataset[j][0])]['vector'].values[0]
    return dl1, dl2


def make_pairFASTA(dataset, itr1, outpath):
    '''Construct the ith ncRNA and the remaining unmatched multi-view features.

    According to the process shown in the paper (alignment sequence, extraction of features, etc.), the ith ncRNA and the remaining unmatched multi-perspective features were constructed.

    Args:

        dataset: All the ncRNA data.

        itr1: The ith a ncRNA.

        outpath: Feature storage path.

    Returns:

    .. note::

        This is the multithreaded version, so you should be careful when debugging.

    '''
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

def main(infile, outpath):
    start = datetime.datetime.now()

    itr1 = 0

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='make_profile:')
    parser.add_argument('--infile', '-i',
                        default='../data/sequence_all.fa',
                        help='Fasta files containing ncRNA sequences')
    parser.add_argument('--outpath', '-o',
                        default='../features/profile/',
                        help='Output paths for multi-view features')
    args = parser.parse_args()

    main(args.infile, args.outpath)