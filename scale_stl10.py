import argparse
import numpy as np
import chainer
from chainer import cuda
from chainer import Variable
import chainer.functions as F
import cupy as xp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stlpath', type=str, default="training_data/STL96.npy")
    return parser.parse_args()


def resize(stl_path, size):
    batchsize = 1000
    stl_96_cpu = np.load(stl_path).astype(np.float32)
    for n in range(0, 100000//batchsize):
        print(n*batchsize, (n+1)*batchsize)
        stl_96_gpu = Variable(cuda.to_gpu(stl_96_cpu[n*batchsize:(n+1)*batchsize]))
        if size == 48:
            x = F.average_pooling_2d(stl_96_gpu, 2).data
        elif size == 32:
            x = F.resize_images(stl_96_gpu, (32, 32)).data
        else:
            raise ValueError('Size should be 48 or 32!')
        if n==0:
            stl_resized_cpu = cuda.to_cpu(x)
        else:
            stl_resized_cpu = np.concatenate([stl_resized_cpu, cuda.to_cpu(x)], axis=0)
    return stl_resized_cpu


if __name__ == '__main__':
    args = parse_args()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    stl_resized = resize(args.stlpath, 48)
    print("saved shape:", stl_resized.shape)
    np.save("training_data/STL48.npy", stl_resized)

    stl_resized = resize(args.stlpath, 32)
    print("saved shape:", stl_resized.shape)
    np.save("training_data/STL32.npy", stl_resized)