"""
Source: https://github.com/AkinoriTanaka-phys/DOT/blob/master/evaluation.py

Copyright (c) 2019 AkinoriTanaka-phys
Released under the MIT license
https://github.com/AkinoriTanaka-phys/DOT/blob/master/LICENSE

Copyright (c) 2017 pfnet-research
Released under the MIT license
https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE
"""

import numpy as np
import scipy


import chainer
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from inception_score import Inception
from inception_score import inception_score

import math


def load_inception_model():
    model = Inception()
    serializers.load_hdf5('metric/inception_score.model', model)
    model.to_gpu()
    return model


def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp
    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        # print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False),\
             chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        ys[batch_start:batch_end] = y.data

    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    # cov = F.cross_covariance(ys, ys, reduce="no").data.get()
    cov = np.cov(chainer.cuda.to_cpu(ys).T)

    return mean, cov


def FID(m0, c0, m1, c1):
    ret = 0
    ret += np.sum((m0-m1) ** 2)
    ret += np.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)


def calc_FID(img, model, data='CIFAR'):
    """Frechet Inception Distance proposed by
    https://arxiv.org/abs/1706.08500"""
    data_m = np.load("metric/{}_inception_mean.npy".format(data))
    data_c = np.load("metric/{}_inception_cov.npy".format(data))

    with chainer.using_config('train', False),\
         chainer.using_config('enable_backprop', False):
        mean, cov = get_mean_cov(model, img)
    fid = FID(data_m, data_c, mean, cov)
    return fid


def calc_inception_score(img, model):
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        mean, std = inception_score(model, img)
    return mean.get().item(), std.get().item()
