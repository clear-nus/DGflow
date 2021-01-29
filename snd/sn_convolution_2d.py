# from https://github.com/pfnet-research/chainer-gan-lib/tree/master/common/sn
# Copyright (c) 2017 pfnet-research
# Released under the MIT license
# https://github.com/pfnet-research/chainer-gan-lib/blob/master/LICENSE
import math
import numpy as np
import chainer
from chainer import cuda
from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.functions.array.broadcast import broadcast_to
import snd.max_sv as max_sv


class SNConvolution2D(Convolution2D):
    """Two-dimensional convolutional layer with spectral normalization.
    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.
    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        use_gamma (bool): If true, apply scalar multiplication to the 
            normalized weight (i.e. reparameterize).
        Ip (int): The number of power iteration for calculating the spcetral 
            norm of the weights. 
    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.
    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Normalized (Reparametrized) weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias)

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
        super(SNConvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad)
