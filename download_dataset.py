"""Source:
https://github.com/AkinoriTanaka-phys/DOT/blob/master/load_dataset.py

Copyright (c) 2019 AkinoriTanaka-phys
Released under the MIT license
https://github.com/AkinoriTanaka-phys/DOT/blob/master/LICENSE
"""

from torchvision import datasets, transforms
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--root', type=str, default='./data')
    return parser.parse_args()


def load(data, root):
    if data == 'cifar10':
        trainset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        testset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        x_train = trainset.data
        x_test = testset.data
        x = np.transpose(
            np.concatenate([x_train, x_test], axis=0), (0, 3, 1, 2))
        x = x/255
        np.save("training_data/CIFAR.npy", x)

    elif data == 'stl10':
        trainset = datasets.STL10(
            root=root,
            split='unlabeled',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        x = trainset.data
        x = x/255
        np.save("training_data/STL96.npy", x)

    else:
        print("data should be cifar10 or stl10")


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.root):
        os.mkdir(args.root)
    if not os.path.exists("training_data"):
        os.mkdir("training_data")
    load(args.data, args.root)
