import os
import re
import subprocess

import numpy as np
from PIL import Image

__doc__ = """
utils - a utils scripts for GCFM
================================
**utils** is a Python script used to provide generic utility functions.
In this part, there are many functions involving ncRNA sequence, secondary structure, structural subunit, grayscale transformation image construction, pairing probability calculation and so on.

Main Functions
--------------
Here are just a few of the things that **utils** does well:

  - Folder path checking and creation.
  - Call DAFS, RNAfold, structure subunit computing module.
  - Alignment of ncRNA sequence, secondary structure sequence and structural subunit sequence.
  - Base pairing probability image construction, grayscale, alignment and feature extraction.

Usages
------
See Function calls.

Main Program Functions
----------------------
"""

def check_path(path: str):
    '''Check the target path.

    Check if the target path exists and create the path if it does not.

    Args:

        path: Folder path. Not a file path.

    Returns:

        None
    '''

    if not os.path.exists(path):
        os.makedirs(path)


def run_DAFS(path: str):
    ''' Call DAFS to align pairwise sequences.

    Call DAFS to compare the two sequences. The DAFS called by this function is an executable program under Linux.

    Args:

        path: File path. Subject to DAFS, this file must be in the form of only two sequences.

    Returns:

        tuple: Result tuple. This tuple contains two aligned sequences, and the gap is filled with '-'.

    .. note::

            The file contents are as follows:

            >RF00003_ENST00000621107.1

            ATACTTACGTAACAGGAGAAAATACGGCCATGAAGTTGGTGTTTCTCGGGGGCGATTT...

            >RNU6_928P_ENST00000516876.1

            GTGCTCTCTGAAGCAGCACAAATACAAAACTTGGAGTGAAACAGAGATGAG...

            Result like this:

            ATACTTACGTAACAGGAGAAAATACGGCCATGAAGTTGGTGTTTCTCGGGGGCGATTT...

            GTG-CTCT-CT-GA-AGCAGCACAAATACAAA-ACTTGGAGTGAAA-CAG-AGATGAG...
    '''

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
    ''' Call RNAfold to get information of secondary structure and base pairing.

    Call RNAfold to get the secondary structure of the sequence and the probability of base pairing.

    Args:

        path: File path. Contains ncRNA sequences.

    Returns:

        tuple: Result tuple. This tuple contains two secondary structure sequences.

    .. note::

            Result like this:

            ...(((.....(((........))..)).))...

            ..(.....(.(((.....(((........))..)).)).)..)..
    '''
    inn = subprocess.Popen(['../tools-non-py/RNAfold', '-p', '--noPS', path], stdout=subprocess.PIPE, encoding='utf-8')

    inn.wait()
    buf = []
    while True:
        innline = inn.stdout.readline()
        buf.append(innline)
        if not innline:
            break

    return re.split('[\s]', buf[2].strip().upper())[0], re.split('[\s]', buf[8].strip().upper())[0], buf


def get_matrix(input: str, n: int):
    ''' Extract the paired probabilities of nucleotide, and the transformed grayscale image.

        The data was extracted from the files generated by RNAfold, and the probabilities of nucleotide were extracted, as well as the transformed grayscale images.

        Args:

            input: File path. Probability files containing ncRNA nucleotide pairing were generated by RNAfold.

            n: Image size.

        Returns:

            tuple: Result tuple. This tuple contains the paired probabilities of nucleotide, and the transformed grayscale image.

        '''

    il = 40
    ir = 255
    ids = ir - il

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
                ele = re.split('[\s]', line.rstrip(' ubox\n'))
                mat[int(ele[0]) - 1, int(ele[1]) - 1] = float(ele[2]) ** 2
                image_mat[int(ele[0]) - 1, int(ele[1]) - 1] = il + (float(ele[2]) ** 2) * ids
                break
            while (switch_tag == 1) & (tag4 in line):
                ele_image = re.split('[\s]', line.rstrip(' lbox\n'))
                image_mat[int(ele_image[1]) - 1, int(ele_image[0]) - 1] = il + (float(ele_image[2]) ** 2) * ids
                break
            if line == tag1:
                switch_tag = 1
            elif line == tag2:
                break
            else:
                continue

    return mat, image_mat


def cal_bp_pro(mat: list):
    ''' Calculate the probability of left and right and unpaired.

        The data was extracted from the files generated by RNAfold, and the left and right and unpaired probabilities were calculated.

        Args:

            mat: List(array) like, containing ncRNA nucleotide pairing were generated by RNAfold.

        Returns:

            tuple: Result array. This array contains the probability of left and right and unpaired.

    '''
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


def make_8bit(pair: str, ss: str):
    '''The fusion secondary structure transforms the four-bit nucleotide sequence into the eight-bit. (Actually five and nine bit)

    The four-bit nucleotide sequence is converted into an eight-bit sequence according to whether the corresponding position matches.

    Args:

        pair: Nucleotide sequence.

        ss: Secondary structure sequence.

    Returns:

        str: A transformed eight-bit nucleotide sequence.
    '''
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


def save_image(dataset: list, image_mat: list, output_path: str):
    '''Save the transformed grayscale image for use by the pre-training model.

    The gray image is obtained according to the transformed image matrix, and the features are extracted with the pre-training model after saving.

    Args:

        dataset: A list (array like) containing all the data.

        image_mat: A list (array like) containing image data.

        output_path: The output path.

    Returns:

        str: A symbol that indicates whether or not it is complete.
    '''
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


def wk_image(output_dir:str, image_dir:str):
    '''Feature extraction using pre-trained image network.

    Use the pre-trained image network screened by multiple classification ability to extract features.

    Args:

        output_dir: Feature output path.

        image_dir: Grayscale image path.

    Returns:

        str: A symbol that indicates whether or not it is complete.
    '''
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
    '''Calculate the structural subunit.

    Args:

        centroid_struct: Secondary structure sequence.

        pss_path: The path of a dynamic programming program.

    Returns:

        str: A sequence of structural subunits.

    .. note::

        Dynamic programming program from https://github.com/cookkate/rnascan
    '''
    import tempfile
    import subprocess

    temphandle = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
    temphandle2 = tempfile.NamedTemporaryFile(delete=False, mode='w+t')

    temphandle.write(centroid_struct)
    temphandle.close()

    parse_args = [pss_path, temphandle.name, temphandle2.name]
    parse_structure_proc = subprocess.Popen(parse_args)
    parse_structure_proc.wait()

    annotated_struct = temphandle2.readline().rstrip('\n')
    return annotated_struct


def align_anno_seq(anno_struct: str, seq: str) -> str:
    '''Align the structure subunits sequences.

    Use alignment information of nucleotide sequences, alignment of structural subunits sequences.

    Args:

        anno_struct: A sequence of structural subunits.

        seq: Nucleotide sequence.

    Returns:

        str: The sequence of structural subunits after alignment.
    '''
    for i in range(len(seq)):
        if seq[i] == '-':
            anno_struct = anno_struct[:i] + '-' + anno_struct[i:]
    assert len(anno_struct) == len(seq)
    return anno_struct
