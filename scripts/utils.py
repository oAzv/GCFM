import os
import subprocess
import re
import numpy as np
from PIL import Image


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_DAFS(path):

    inn = subprocess.Popen(['../tools-non-py/dafs', path], stdout=subprocess.PIPE, encoding='utf-8')

    inn.wait()
    # a = inn.communicate()
    buf = []
    while True:
        # sys.stdout.flush()
        innline = inn.stdout.readline()
        buf.append(innline)
        if not innline:
            break

    return buf[4].strip().upper(), buf[6].strip().upper()


def run_RNAfold(path):
    inn = subprocess.Popen(['../tools-non-py/RNAfold', '-p', '--noPS', path], stdout=subprocess.PIPE, encoding='utf-8')

    inn.wait()
    buf = []
    while True:
        innline = inn.stdout.readline()
        # print(len(innline))
        buf.append(innline)
        # print(buf2)
        if not innline:
            break

    return re.split('[\s]', buf[2].strip().upper())[0], re.split('[\s]', buf[8].strip().upper())[0], buf


def get_matrix(input, n):
    il = 40
    ir = 255
    ids = ir - il

    mat = np.zeros((n, n), dtype=np.float32)
    image_mat = np.zeros((n, n), dtype=np.float32)
    # image_mat = image_mat + ir
    # arr = []
    tag1 = '%start of base pair probability data\n'
    tag2 = 'showpage\n'
    tag3 = 'ubox'
    tag4 = 'lbox'

    switch_tag = 0

    with open(input) as f:
        for line in f:
            while (switch_tag == 1) & (tag3 in line):
                ele = re.split('[\s]', line.rstrip(' ubox\n'))
                # arr = arr + [[ele[0], ele[1], float(ele[2]) ** 2]]
                mat[int(ele[0]) - 1, int(ele[1]) - 1] = float(ele[2]) ** 2
                # image_mat[int(ele[0]) - 1, int(ele[1]) - 1] = (float(ele[2]) ** 2) * 255
                image_mat[int(ele[0]) - 1, int(ele[1]) - 1] = il + (float(ele[2]) ** 2) * ids
                # image_mat[int(ele[0]) - 1, int(ele[1]) - 1] = ir - range_change(float(ele[2])) * ir
                break
            while (switch_tag == 1) & (tag4 in line):
                ele_image = re.split('[\s]', line.rstrip(' lbox\n'))
                # image_mat[int(ele_image[1]) - 1, int(ele_image[0]) - 1] = (float(ele_image[2]) ** 2) * 255
                image_mat[int(ele_image[1]) - 1, int(ele_image[0]) - 1] = il + (float(ele_image[2]) ** 2) * ids
                # image_mat[int(ele_image[1]) - 1, int(ele_image[0]) - 1] = ir - range_change(float(ele_image[2])) * ir
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


def save_image(dataset, image_mat, output_path):
    len_pair1 = len(dataset[0][1])
    len_pair2 = len(dataset[1][1])

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

    check_path(output_path)

    image1.save(output_path + str(dataset[0][0]) + '.png')
    image2.save(output_path + str(dataset[1][0]) + '.png')
    return 'Done'


def wk_image(output_dir, image_dir):
    from torchvision import models
    from scripts.WK_NetArch import alexnet_features, resnet101_features, vgg16_features
    from scripts.WK_NetArch import wk_tools as wkt

    files_list = wkt.get_image(image_dir)

    def extract_single(tag, output_dir, files_list):
        if tag == 'alexnet':
            alexnet = models.alexnet(pretrained=True)
            model = alexnet_features.EncoderCNN(alexnet)
        elif tag == 'resnet101':
            resnet101 = models.alexnet(pretrained=True)
            model = resnet101_features.EncoderCNN(resnet101)
        elif tag == 'vgg16':
            vgg16 = models.vgg16(pretrained=True)
            model = vgg16_features.EncoderCNN(vgg16)
        wkt.extract_features(model, tag, output_dir, files_list)

    wkn_tags = ['alexnet', 'resnet101', 'vgg16']
    for tag in wkn_tags:
        extract_single(tag, output_dir, files_list)
    return 'Done'

def get_annotated_struct(centroid_struct: str, pss_path: str) -> str:
    import tempfile
    import subprocess

    temphandle = tempfile.NamedTemporaryFile(delete=False, mode='w+t')  # for centroid structure
    temphandle2 = tempfile.NamedTemporaryFile(delete=False, mode='w+t')

    # centroid_struct = '(((((((....)).)))))................(((((((((.............((((.....)))).((((........))))..)))))))))......'

    temphandle.write(centroid_struct)
    temphandle.close()

    parse_args = [pss_path, temphandle.name, temphandle2.name]
    # print parse_args
    parse_structure_proc = subprocess.Popen(parse_args)
    parse_structure_proc.wait()

    annotated_struct = temphandle2.readline().rstrip('\n')
    return annotated_struct


def align_anno_seq(anno_struct: str, seq: str) -> str:
    # print(len(anno_struct), len(seq))
    for i in range(len(seq)):
        if seq[i] == '-':
            anno_struct = anno_struct[:i] + '-' + anno_struct[i:]
    # print(len(anno_struct),len(seq))
    assert len(anno_struct) == len(seq)
    return anno_struct
