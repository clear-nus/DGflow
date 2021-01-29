"""Source:
https://github.com/AkinoriTanaka-phys/DOT/blob/master/get_mean_cov_2048featurespace.py

Copyright (c) 2019 AkinoriTanaka-phys
Released under the MIT license
https://github.com/AkinoriTanaka-phys/DOT/blob/master/LICENSE
"""

import argparse
import chainer
from chainer import cuda
from chainer import serializers

from inception_score import Inception

import cupy as xp
import numpy as np
from evaluation import get_mean_cov


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='CIFAR')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = Inception()
    serializers.load_hdf5('metric/inception_score.model', model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    datapath = 'training_data/{}.npy'.format(args.data)
    mean_savepath = 'metric/{}_inception_mean.npy'.format(args.data)
    cov_savepath = 'metric/{}_inception_cov.npy'.format(args.data)

    img = 255*xp.load(datapath).astype(xp.float32)
    with chainer.using_config('train', False),\
         chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, img)

    np.save(mean_savepath, mean)
    np.save(cov_savepath, cov)
