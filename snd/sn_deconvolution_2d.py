## modified version
# Copyright (c) 2017 pfnet-research
# Released under the MIT license
# https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE
import math
import numpy as np
import chainer
from chainer import cuda
from chainer.functions.connection import deconvolution_2d
from chainer import initializers
from chainer import link
from chainer.links.connection.deconvolution_2d import Deconvolution2D
from chainer.functions.array.broadcast import broadcast_to
import snd.max_sv as max_sv

class SNDeconvolution2D(Deconvolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, outsize=None, initialW=None, initial_bias=None, use_gamma=False, Ip=1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNDeconvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, outsize, initialW, initial_bias)

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        xp = cuda.get_array_module(self.W.data)
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_sv.max_singular_value(W_mat, self.u, self.Ip)
        sigma = broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        self.u = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNDeconvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deconvolution_2d.deconvolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad, self.outsize)


