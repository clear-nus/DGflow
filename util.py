import torch
import torch.nn as nn
from chainer import serializers

import chainer_models
import torch_models
import gen_models.resnet_32


def load_model(model_type, net_type, model_path, config):
    tM = None
    if model_type == 'gen':
        if net_type == 'c_dcgan':
            cG = chainer_models.DCGANGenerator(
                bottom_width=config['bottom_width'])
            serializers.load_npz(model_path, cG)
            cG = cG.to_gpu()
            tM = torch_models.DCGANGenerator(cG, bw=config['bottom_width'])
        elif net_type == 'c_resnet':
            cG = gen_models.resnet_32.ResNetGenerator()
            serializers.load_npz(model_path, cG)
            cG = cG.to_gpu()
            tM = torch_models.ResNetGenerator32(cG)
        elif net_type == 't_cnn':
            tM = torch_models.CNNDecoder(
                isize=32, nc=3, k=config['z_dim'],
                act=nn.Tanh(), scale_image=config['image_size'])
            tM.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu')))
        elif net_type == 't_cnn_vae':
            tM = torch_models.CNNDecoder(
                isize=32, nc=3, k=config['z_dim'],
                act=torch.nn.Sigmoid(), scale_image=config['image_size'])
            tM.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu')))

    elif model_type == 'disc':
        if net_type == 'c_sndcgan':
            cD = chainer_models.SNDCGANDiscriminator(
                bottom_width=config['bottom_width'])
            serializers.load_npz(model_path, cD)
            cD = cD.to_gpu()
            tM = torch_models.SNDCGANDiscriminator(
                cD, bw=config['bottom_width'])
        elif net_type == 'c_wgan':
            cD = chainer_models.WGANDiscriminator(
                bottom_width=config['bottom_width'])
            serializers.load_npz(model_path, cD)
            cD = cD.to_gpu()
            tM = torch_models.WGANDiscriminator(cD, bw=config['bottom_width'])
    elif model_type == 'corr':
        if net_type == 't_sndcgan':
            tM = torch_models.SNDCGANDiscriminator(bw=config['bottom_width'])
            tM.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise ValueError('model_type can only be one of [gen, disc, corr]')
    return tM
