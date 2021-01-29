import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
from chainer2torch import (
    copy_Linear,
    copy_ConvTranspose2d,
    copy_BatchNorm2d,
    copy_Conv2d,
    copy_CategoricalConditionalBatchNorm2d)


def _upsample(x):
    h, w = x.shape[2:]
    return nn.functional.interpolate(x, size=(h * 2, w * 2), mode='nearest')


def upsample_conv(x, conv):
    return conv(_upsample(x))


class DCGANGenerator(nn.Module):
    def __init__(self, G, n_hidden=128, bw=4, ch=512):
        super().__init__()
        self.ch = ch
        self.bw = bw
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.l0 = nn.Linear(n_hidden, bw*bw*ch)
        copy_Linear(self.l0, G.l0)
        self.dc1 = nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1)
        copy_ConvTranspose2d(self.dc1, G.dc1)
        self.dc2 = nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1)
        copy_ConvTranspose2d(self.dc2, G.dc2)
        self.dc3 = nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1)
        copy_ConvTranspose2d(self.dc3, G.dc3)
        self.dc4 = nn.ConvTranspose2d(ch // 8, 3, 3, 1, 1)
        copy_ConvTranspose2d(self.dc4, G.dc4)
        self.bn0 = nn.BatchNorm2d(bw * bw * ch, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.bn0, G.bn0)
        self.bn1 = nn.BatchNorm2d(ch // 2, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.bn1, G.bn1)
        self.bn2 = nn.BatchNorm2d(ch // 4, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.bn2, G.bn2)
        self.bn3 = nn.BatchNorm2d(ch // 8, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.bn3, G.bn3)

    def forward(self, z, **kwargs):
        h = self.l0(z)
        h = h.view(-1, self.ch * self.bw * self.bw, 1, 1)
        h = self.relu(self.bn0(h))
        h = h.view(-1, self.ch, self.bw, self.bw)
        h = self.relu(self.bn1(self.dc1(h)))
        h = self.relu(self.bn2(self.dc2(h)))
        h = self.relu(self.bn3(self.dc3(h)))
        o = self.tanh(self.dc4(h))
        return o


class WGANDiscriminator(nn.Module):
    def __init__(self, D, bw=4, ch=512, output_dim=1):
        super().__init__()
        c0 = nn.Conv2d(3, ch // 8, 3, 1, 1)
        copy_Conv2d(c0, D.c0)
        self.c0 = c0

        c1 = nn.Conv2d(ch // 8, ch // 4, 4, 2, 1)
        copy_Conv2d(c1, D.c1)
        self.c1 = c1

        c1_0 = nn.Conv2d(ch // 4, ch // 4, 3, 1, 1)
        copy_Conv2d(c1_0, D.c1_0)
        self.c1_0 = c1_0

        c2 = nn.Conv2d(ch // 4, ch // 2, 4, 2, 1)
        copy_Conv2d(c2, D.c2)
        self.c2 = c2

        c2_0 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        copy_Conv2d(c2_0, D.c2_0)
        self.c2_0 = c2_0

        c3 = nn.Conv2d(ch // 2, ch // 1, 4, 2, 1)
        copy_Conv2d(c3, D.c3)
        self.c3 = c3

        c3_0 = nn.Conv2d(ch // 1, ch // 1, 3, 1, 1)
        copy_Conv2d(c3_0, D.c3_0)
        self.c3_0 = c3_0

        l4 = nn.Linear(bw * bw * ch, output_dim)
        copy_Linear(l4, D.l4)
        self.l4 = l4

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        h = self.lrelu(self.c0(x))
        h = self.lrelu(self.c1(h))
        h = self.lrelu(self.c1_0(h))
        h = self.lrelu(self.c2(h))
        h = self.lrelu(self.c2_0(h))
        h = self.lrelu(self.c3(h))
        h = self.lrelu(self.c3_0(h))
        h = h.view(x.size(0), -1)
        return self.l4(h)


class SNDCGANDiscriminator(nn.Module):
    def __init__(self, D=None, bw=4, ch=512, output_dim=1):
        super().__init__()
        c0_0 = nn.Conv2d(3, ch // 8, 3, 1, 1)
        if D is not None:
            copy_Conv2d(c0_0, D.c0_0)
        self.c0_0 = spectral_norm(c0_0)

        c0_1 = nn.Conv2d(ch // 8, ch // 4, 4, 2, 1)
        if D is not None:
            copy_Conv2d(c0_1, D.c0_1)
        self.c0_1 = spectral_norm(c0_1)

        c1_0 = nn.Conv2d(ch // 4, ch // 4, 3, 1, 1)
        if D is not None:
            copy_Conv2d(c1_0, D.c1_0)
        self.c1_0 = spectral_norm(c1_0)

        c1_1 = nn.Conv2d(ch // 4, ch // 2, 4, 2, 1)
        if D is not None:
            copy_Conv2d(c1_1, D.c1_1)
        self.c1_1 = spectral_norm(c1_1)

        c2_0 = nn.Conv2d(ch // 2, ch // 2, 3, 1, 1)
        if D is not None:
            copy_Conv2d(c2_0, D.c2_0)
        self.c2_0 = spectral_norm(c2_0)

        c2_1 = nn.Conv2d(ch // 2, ch // 1, 4, 2, 1)
        if D is not None:
            copy_Conv2d(c2_1, D.c2_1)
        self.c2_1 = spectral_norm(c2_1)

        c3_0 = nn.Conv2d(ch // 1, ch // 1, 3, 1, 1)
        if D is not None:
            copy_Conv2d(c3_0, D.c3_0)
        self.c3_0 = spectral_norm(c3_0)

        l4 = nn.Linear(bw * bw * ch, output_dim)
        if D is not None:
            copy_Linear(l4, D.l4)
        self.l4 = spectral_norm(l4)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        h = self.lrelu(self.c0_0(x))
        h = self.lrelu(self.c0_1(h))
        h = self.lrelu(self.c1_0(h))
        h = self.lrelu(self.c1_1(h))
        h = self.lrelu(self.c2_0(h))
        h = self.lrelu(self.c2_1(h))
        h = self.lrelu(self.c3_0(h))
        h = h.view(x.size(0), -1)
        return self.l4(h)


class ResidualBlock(nn.Module):
    def __init__(self, B, in_channels, out_channels,
                 hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super().__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels\
            if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        c1 = nn.Conv2d(in_channels, hidden_channels, ksize, padding=pad)
        copy_Conv2d(c1, B.c1)
        self.c1 = c1
        c2 = nn.Conv2d(hidden_channels, out_channels, ksize, padding=pad)
        copy_Conv2d(c2, B.c2)
        self.c2 = c2
        if n_classes > 0:
            b1 = CategoricalConditionalBatchNorm2d(
                n_classes, in_channels, eps=2e-5, momentum=0.1)
            copy_CategoricalConditionalBatchNorm2d(b1, B.b1)
            self.b1 = b1
            b2 = CategoricalConditionalBatchNorm2d(
                n_classes, hidden_channels, eps=2e-5, momentum=0.1)
            copy_CategoricalConditionalBatchNorm2d(b2, B.b2)
            self.b2 = b2
        else:
            b1 = nn.BatchNorm2d(in_channels, eps=2e-5, momentum=0.1)
            copy_BatchNorm2d(b1, B.b1)
            self.b1 = b1
            b2 = nn.BatchNorm2d(hidden_channels, eps=2e-5, momentum=0.1)
            copy_BatchNorm2d(b2, B.b2)
            self.b2 = b2
        if self.learnable_sc:
            c_sc = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)
            copy_Conv2d(c_sc, B.c_sc)
            self.c_sc = c_sc

    def residual(self, x, y=None, **kwargs):
        h = x
        if kwargs.get('resize') is not None:
            del kwargs['resize']
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None, z=None, **kwargs):
        return self.residual(x, y, **kwargs) + self.shortcut(x)


class ResNetGenerator128(nn.Module):
    def __init__(self, G, ch=64, dim_z=128, bottom_width=4,
                 activation=nn.ReLU(), n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        copy_Linear(self.l1, G.l1)
        self.block2 = ResidualBlock(G.block2, ch * 16, ch * 16,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block3 = ResidualBlock(G.block3, ch * 16, ch * 8,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block4 = ResidualBlock(G.block4, ch * 8, ch * 4,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block5 = ResidualBlock(G.block5, ch * 4, ch * 2,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block6 = ResidualBlock(G.block6, ch * 2, ch * 1,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.b7 = nn.BatchNorm2d(ch, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.b7, G.b7)
        self.l7 = nn.Conv2d(ch, 3, 3, padding=1)
        copy_Conv2d(self.l7, G.l7)
        self.tanh = nn.Tanh()

    def forward(self, z, y=None, **kwargs):
        h = z
        h = self.l1(h)
        h = h.view((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)
        h = self.b7(h)
        h = self.activation(h)
        h = self.tanh(self.l7(h))
        return h


class ResNetGenerator64(nn.Module):
    def __init__(self, G, ch=64, dim_z=128, bottom_width=4,
                 activation=nn.ReLU(), n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
        copy_Linear(self.l1, G.l1)
        self.block2 = ResidualBlock(G.block2, ch * 16, ch * 8,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block3 = ResidualBlock(G.block3, ch * 8, ch * 4,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block4 = ResidualBlock(G.block4, ch * 4, ch * 2,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block5 = ResidualBlock(G.block5, ch * 2, ch * 1,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.b6 = nn.BatchNorm2d(ch, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.b6, G.b6)
        self.l6 = nn.Conv2d(ch, 3, 3, padding=1)
        copy_Conv2d(self.l6, G.l6)
        self.tanh = nn.Tanh()

    def forward(self, z, y=None, **kwargs):
        h = z
        h = self.l1(h)
        h = h.view((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.b6(h)
        h = self.activation(h)
        h = self.tanh(self.l6(h))
        return h


class ResNetGenerator32(nn.Module):
    def __init__(self, G, ch=256, dim_z=128, bottom_width=4,
                 activation=nn.ReLU(), n_classes=0):
        super().__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        copy_Linear(self.l1, G.l1)
        self.block2 = ResidualBlock(G.block2, ch, ch,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block3 = ResidualBlock(G.block3, ch, ch,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.block4 = ResidualBlock(G.block4, ch, ch,
                                    activation=activation,
                                    upsample=True,
                                    n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(ch, eps=2e-5, momentum=0.1)
        copy_BatchNorm2d(self.b5, G.b5)
        self.c5 = nn.Conv2d(ch, 3, 3, padding=1)
        copy_Conv2d(self.c5, G.c5)
        self.tanh = nn.Tanh()

    def forward(self, z, y=None, **kwargs):
        h = z
        h = self.l1(h)
        h = h.view((h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.b5(h)
        h = self.activation(h)
        h = self.tanh(self.c5(h))
        return h


class CNNDecoder(nn.Module):
    # Source:
    # https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    def __init__(self, isize, nc, k=100, ngf=64, act=nn.Tanh(), scale_image=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(k, cngf),
                        nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module(
            'initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1,
                                               bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-act'.format(nc),
                        act)
        self.act = act
        self.main = main
        self.scale_image = 0 if scale_image == isize else scale_image

    def forward(self, input, resize=True):
        s = input.shape
        input = input.view(s[0], s[1], 1, 1)
        output = self.main(input)
        if type(self.act) == torch.nn.Sigmoid:
            # Convert [0, 1] to [-1, 1]
            output = (output - 0.5) * 2
        if self.scale_image > 0 and resize:
            output = F.interpolate(
                output, size=self.scale_image, mode='bilinear',
                align_corners=True)
        return output
