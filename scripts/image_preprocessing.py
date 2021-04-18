#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from Bio import SeqIO
import os
from PIL import Image
import subprocess
import pandas as pd
import numpy as np

from torch import optim
from torchvision import models, transforms
from WK_NetArch import wk_tools as wkt
from WK_NetArch import alexnet_features, resnet101_features, vgg16_features
from scripts.utils import run_DAFS, run_RNAfold, make_8bit, get_matrix

import argparse

__doc__ = """
image_preprocessing - a scripts for image feature construction
===========================================================================

**image_preprocessing** is a Python script that provides function to process image and construct image feature.

Main Functions
--------------
Here are just a few of the things that **image_preprocessing** does well:

  - Generate grayscale image.
  - Construct image feature.

Main Program Functions
----------------------

"""

def extract_single(tag, features_dir, files_list):
    if tag == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        model = alexnet_features.EncoderCNN(alexnet)
    elif tag == 'resnet101':
        resnet101 = models.alexnet(pretrained=True)
        model = resnet101_features.EncoderCNN(resnet101)
    elif tag == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
        model = vgg16_features.EncoderCNN(vgg16)

    wkt.extract_features(model, tag, features_dir, files_list)

def make_image(dataset, itr1, outpath):
    for j in range(itr1 + 1, len(dataset)):
        len_pair1 = len(dataset[itr1][2])
        len_pair2 = len(dataset[j][2])

        path_to_pairFasta = "./pair" + str(itr1) + "," + str(j) + ".fa"
        pairFa = ""
        for k in [dataset[itr1], dataset[j]]:
            pairFa += ">" + k[0] + "\n" + k[2] + "\n"
        with open(path_to_pairFasta, 'w') as f:
            f.write(pairFa)

        pair1, pair2 = run_DAFS(path_to_pairFasta)
        ss1, ss2 = run_RNAfold(path_to_pairFasta)

        pair1 = make_8bit(pair1, ss1)
        pair2 = make_8bit(pair2, ss2)

        subprocess.call(["rm", path_to_pairFasta])

        dp1 = './' + str(dataset[itr1][0]) + '_dp.ps'
        dp2 = './' + str(dataset[j][0]) + '_dp.ps'

        mat1, mat2 = get_matrix(dp1, len(pair1))

        image_mat = [mat2]

        mat1, mat2 = get_matrix(dp2, len(pair2))

        image_mat = image_mat + [mat2]

        if itr1 == 0:
            image1 = Image.fromarray(image_mat[0]).convert('L')
            image2 = Image.fromarray(image_mat[1]).convert('L')

            re_box = (0, 0, len_pair2, len_pair2) if len_pair1 > len_pair2 else (0, 0, len_pair1, len_pair1)
            re_image = image2.crop(re_box) if len_pair1 > len_pair2 else image1.crop(re_box)
            if len_pair1 > len_pair2:
                image2 = re_image
            else:
                image1 = re_image

            rx = 256
            ry = rx
            re_size = (rx, ry)

            if image1.size > re_size:
                image1 = image1.resize(re_size, Image.ANTIALIAS)
            else:
                image1 = image1.resize(re_size, Image.BICUBIC)

            if image2.size > re_size:
                image2 = image2.resize(re_size, Image.ANTIALIAS)
            else:
                image2 = image2.resize(re_size, Image.BICUBIC)

            image1 = image1.convert('RGB')
            image2 = image2.convert('RGB')

            image1.save(outpath + str(dataset[itr1][0]) + '.png')
            image2.save(outpath + str(dataset[j][0]) + '.png')

def wk_main(infile, data_dir, features_dir):

    dataset = []
    wkt.check_path(features_dir)
    files_list = wkt.get_image(data_dir)
    wkn_tags = ['alexnet', 'resnet101', 'vgg16']

    for record in SeqIO.parse(infile, "fasta"):

        id_part = record.id
        id_parts = id_part.split(",")
        dataset = dataset + [[id_parts[0], int(id_parts[1])]]

    make_image(dataset, 0, data_dir)
    os.system('rm *.ps')

    # use_gpu = torch.cuda.is_available()
    for tag in wkn_tags:
        extract_single(tag, features_dir, files_list)

if __name__ == '__main__':

    data_dir = '../features/image/'
    features_dir = '../features/image_features/'
    infile = '../data/sequence_all.fa'

    parser = argparse.ArgumentParser(description='image_preprocessing:')
    parser.add_argument('--infile', '-i',
                        default='../data/sequence_all.fa',
                        help='Fasta files containing ncRNA sequences')
    parser.add_argument('--data', '-d',
                        default='../features/image/',
                        help='The paths of images')
    parser.add_argument('--outpath', '-o',
                        default='../features/image_features/',
                        help='Output paths of features')

    args = parser.parse_args()

    wk_main(args.infile, args.data_dir, args.outpath)