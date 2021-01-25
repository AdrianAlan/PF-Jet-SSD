import setGPU

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import yaml

from ssd.net import build_ssd
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Measure SSD's Inference Time")
    parser.add_argument('model', type=str, help='Input model name')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    base = '{}/{}'.format(config['output']['model'], args.model)
    source_path = '{}.pth'.format(base)
    logger = set_logging('Test_SSD', '{}.log'.format(base), args.verbose)
    logger.info('Testing {0} model'.format(source_path))

    if not torch.cuda.is_available():
        logger.error('CUDA not available. Aborting!')
        sys.exit(0)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    inference_pref = config['inference_pref']
    batch_sizes = inference_pref['batch_sizes']
    samples = inference_pref['samples']
    rank = inference_pref['rank']

    ssd_settings = config['ssd_settings']
    ssd_settings['n_classes'] += 1
    in_dim = ssd_settings['input_dimensions']

    logger.info('Initiating...')

    device = torch.device('cuda')
    net = build_ssd(rank, ssd_settings, inference=True)
    net.load_weights(source_path)
    net.eval()
    net.to(device)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    batch_sizes = np.array(batch_sizes)
    bs = len(batch_sizes)
    lat_means, thr_means = (np.zeros((bs)), np.zeros((bs)))
    lat_stds, thr_stds = (np.zeros((bs)), np.zeros((bs)))

    for x, bs in enumerate(batch_sizes):
        logger.info('Measure for batch size: {0}'.format(bs))

        dummy_input = torch.unsqueeze(torch.randn(in_dim), 0)
        batch = torch.cat(bs*[dummy_input]).to(device)

        logger.info('GPU warm-up')
        for _ in range(10):
            _ = net(batch)

        logger.info('Measuring latency')
        measurements = np.zeros((samples))
        with torch.no_grad():
            for i in range(samples):
                s.record()
                _ = net(batch)
                e.record()
                torch.cuda.synchronize()
                measurements[i] = s.elapsed_time(e)

        lat_mean = np.mean(measurements / bs)
        lat_std = np.std(measurements / bs)
        thr_mean = np.mean(1000*bs / measurements)
        thr_std = np.std(1000*bs / measurements)

        lat_means[x] = lat_mean
        lat_stds[x] = lat_std
        thr_means[x] = thr_mean
        thr_stds[x] = thr_std

        logger.info('Latency: {0:.2f} ± {1:.2f} ms'.format(lat_mean, lat_std))
        logger.info('Throughput: {0:.2f} eps'.format(thr_mean))

    logger.info('Plotting results')
    plot = Plotting(save_dir=config['output']['plots'])
    plot.draw_errorbar(batch_sizes, lat_means, lat_stds,
                       'Latency [ms]', 'latency-gpu')
    plot.draw_errorbar(batch_sizes, thr_means, thr_stds,
                       'Throughput [eps]', 'throughput-gpu')
