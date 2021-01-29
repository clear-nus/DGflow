""" Contains code to copy chainer weights to pytorch.
"""

import torch
import cupy as xp


def _to_numpy(arr):
    if type(arr) == xp.core.core.ndarray:
        np_arr = arr.get()
    elif type(arr.data) == xp.core.core.ndarray:
        np_arr = arr.data
        np_arr = np_arr.get()
    else:
        np_arr = arr.data.copy()
    return np_arr


def copy_weights(to, fr):
    to.data.copy_(torch.from_numpy(_to_numpy(fr)))


def copy_Linear(torch_layer, chainer_layer):
    copy_weights(torch_layer.weight, chainer_layer.W)
    copy_weights(torch_layer.bias, chainer_layer.b)


def copy_ConvTranspose2d(torch_layer, chainer_layer):
    copy_weights(torch_layer.weight, chainer_layer.W)
    copy_weights(torch_layer.bias, chainer_layer.b)


def copy_Conv2d(torch_layer, chainer_layer):
    copy_weights(torch_layer.weight, chainer_layer.W)
    copy_weights(torch_layer.bias, chainer_layer.b)


def copy_BatchNorm2d(torch_layer, chainer_layer):
    copy_weights(torch_layer.weight, chainer_layer.gamma)
    copy_weights(torch_layer.bias, chainer_layer.beta)
    torch_layer.running_mean.data.copy_(
        torch.from_numpy(_to_numpy(chainer_layer.avg_mean)))
    torch_layer.running_var.data.copy_(
        torch.from_numpy(_to_numpy(chainer_layer.avg_var)))
    torch_layer.num_batches_tracked.data.copy_(torch.tensor(chainer_layer.N))


def copy_Embedding(torch_layer, chainer_layer):
    copy_weights(torch_layer.weight, chainer_layer.W)


def copy_CategoricalConditionalBatchNorm2d(torch_layer, chainer_layer):
    copy_Embedding(torch_layer.weights, chainer_layer.gammas)
    copy_Embedding(torch_layer.biases, chainer_layer.betas)
    torch_layer.running_mean.data.copy_(
        torch.from_numpy(_to_numpy(chainer_layer.avg_mean)))
    torch_layer.running_var.data.copy_(
        torch.from_numpy(_to_numpy(chainer_layer.avg_var)))
    torch_layer.num_batches_tracked.data.copy_(torch.tensor(chainer_layer.N))
