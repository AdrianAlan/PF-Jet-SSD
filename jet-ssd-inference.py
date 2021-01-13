import setGPU

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml

from ssd.net import build_ssd
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Measure SSD's Inference Time")
    parser.add_argument('model', type=str, help='Input model name')
    parser.add_argument('config', action=IsValidFile, type=str,
                        help='Path to config file')
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
    measurements = np.zeros((samples, 1))

    for bs in batch_sizes:
        logger.info('Measure for batch size:{0}'.format(bs))

        dummy_input = torch.unsqueeze(torch.randn(in_dim), 0)
        batch = torch.cat(bs*[dummy_input]).to(device)

        logger.info('GPU warm-up')
        for _ in range(10):
            _ = net(batch)

        logger.info('Measuring latency')
        with torch.no_grad():
            for i in range(samples):
                s.record()
                _ = net(batch)
                e.record()
                torch.cuda.synchronize()
                measurements[i] = s.elapsed_time(e)

        mean = np.mean(measurements)
        std = np.std(measurements)

        logger.info('Mean inference: {0:.2f} ± {1:.2f} ms'.format(mean, std))
