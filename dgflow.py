"""
Pytorch implementation of Discriminator Gradient flow (DGflow).
"""

import os
import yaml
import glob
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from chainer import serializers

from evaluation import calc_FID, calc_inception_score
from util import load_model
from inception_score import Inception


def _refine_batch(z, D, G, C, config):
    eta = config['eta']
    f = config['f_div']
    noise_factor = np.sqrt(config['gamma'])
    exp_dir = config['exp_dir']

    def save_images(z, pth):
        pth = os.path.join(exp_dir, 'samples', pth)
        all_imgs = G(z, resize=False).data.cpu().numpy()
        if os.path.exists(pth):
            imgs = np.load(pth)
            all_imgs = np.vstack([imgs, all_imgs])
        np.save(pth, all_imgs)

    def _velocity(z):
        z_t = z.clone()
        z_t.requires_grad_(True)
        if z_t.grad is not None:
            z_t.grad.zero_()
        img_t = G(z_t)
        d_score = D(img_t)
        if C:
            d_score = d_score + C(img_t)

        if f == 'KL':
            s = torch.ones_like(d_score.detach())

        elif f == 'logD':
            s = 1 / (1 + d_score.detach().exp())

        elif f == 'JS':
            s = 1 / (1 + 1 / d_score.detach().exp())

        else:
            raise ValueError()

        s.expand_as(z_t)
        d_score.backward(torch.ones_like(d_score).to(z.device))
        grad = z_t.grad
        return s.data * grad.data

    pth = 'base.npy'
    save_images(z, pth)
    for t in tqdm(range(1, config['steps'] + 1), leave=False):
        v = _velocity(z)
        z = z.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(z)
        if t % config['save_interval'] == 0:
            pth = f'dgflow-step{t}.npy'
            save_images(z, pth)


def refine_batch(D, G, C, config):
    latent_dim = config['z_dim']
    n = config['batch_size']
    noise = torch.randn((n, latent_dim), device='cuda:0')
    _refine_batch(noise, D, G, C, config)


def stabilize_sn(tD, im_size=32, iters=5000):
    pbar = tqdm(range(iters))
    for i in pbar:
        x = torch.rand(10, 3, im_size, im_size).cuda()
        _ = tD(x)


def evaluate_samples(exp_dir, samples_name, evmodel, eval_file_prefix):
    pth = os.path.join(exp_dir, 'samples', samples_name)
    samples = np.load(pth)
    samples = np.asarray(
        np.clip(samples * 127.5 + 127.5, 0.0, 255.0), dtype=np.float32)
    fid = calc_FID(samples, evmodel, data=eval_file_prefix)
    is_mean, is_std = calc_inception_score(samples, evmodel)
    return fid, (is_mean, is_std)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', type=str, required=True, help='path to config file')

    args = parser.parse_args()
    with open(args.config_path, 'r') as fp:
        config = yaml.full_load(fp)

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    exp_dir = os.path.join(
        config['exp_root'], 'dgflow_' + timestamp)
    os.makedirs(os.path.join(exp_dir, 'samples'))
    config['exp_dir'] = exp_dir

    # Load models from checkpoints
    tG = load_model('gen', config['gen_type'], config['gen_path'], config)
    tD = load_model('disc', config['disc_type'], config['disc_path'], config)
    tC = load_model('corr', config['corr_type'], config['corr_path'], config)

    tD, tG = tD.cuda(), tG.cuda()
    if tC:
        tC = tC.cuda()

    # Thermalize spectral norm in the discriminator
    print('[i] Thermalizing spectral norm.')
    stabilize_sn(tD, im_size=config['image_size'])

    # Refine samples
    print('[i] Running DGflow.')
    for k in tqdm(range(0, config['num_imgs'], config['batch_size'])):
        refine_batch(tD, tG, tC, config)

    # Free up GPU memory
    del tG
    del tD
    torch.cuda.empty_cache()

    # Evaluate samples
    print('[i] Running evaluation.')
    evmodel = Inception()
    serializers.load_hdf5('metric/inception_score.model', evmodel)
    evmodel.to_gpu()

    results = {'config': config,
               'FID': dict(),
               'IS': dict()}

    fid, iscore = evaluate_samples(
                    config['exp_dir'],
                    'base.npy',
                    evmodel,
                    config['eval_file_prefix'])
    results['FID']['Base'] = fid.item()
    results['IS']['Base'] = iscore

    for i in tqdm(range(config['save_interval'],
                        config['steps'] + 1,
                        config['save_interval'])):
        fid, iscore = evaluate_samples(
                        config['exp_dir'],
                        f'dgflow-step{i}.npy',
                        evmodel,
                        config['eval_file_prefix'])
        results['FID'][f'Step-{i}'] = fid.item()
        results['IS'][f'Step-{i}'] = iscore

    # Cleanup generated files
    if not config['keep_samples']:
        npy_files = glob.glob(os.path.join(exp_dir, 'samples', '*.npy'))
        for f in npy_files:
            os.remove(f)

    # Save results
    print('[i] Saving results.')
    results_file = os.path.join(exp_dir, 'results.yml')
    with open(results_file, 'w') as fp:
        yaml.dump(results, fp, sort_keys=False)
