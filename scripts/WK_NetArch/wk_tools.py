#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pandas as pd
import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def get_image(data_dir):
    '''Gets the path of all images.

    Args:

        data_dir: Path to images.

    Returns:

        Path of all images.
    '''
    files_list = []
    x = os.walk(data_dir)
    for path, d, filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))
    return files_list


def extractor(net, img_path, use_gpu=True):
    '''Use a pre-training model to extract features.

    Args:

        net: Pre-training model.

        img_path: The path of the image.

        use_gpu: Whether to use GPU or not.

    Returns:

        Extracted features.
    '''
    transform = transforms.Compose([
        transforms.Resize(224),
       transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    if use_gpu == True:
        x = x.cuda()
        net = net.cuda()

    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y


def extract_features(net, tag, features_dir, files_list):
    '''Batch extraction of features.

    Args:

        net: Pre-training model.

        tag: Name of the pre-training model.

        features_dir: Feature output path.

        files_list: Batch image file path.

    Returns:

    '''
    features = []
    fx_path = os.path.join(features_dir, tag + '.pkl')
    for x_path in files_list:
        file_name = x_path.split('/')[-1].strip('.png')
        vector = extractor(net, x_path, use_gpu=False)
        features.append([file_name, vector])
    features = pd.DataFrame(features, columns=['name', 'vector'])
    features.to_pickle(fx_path)

