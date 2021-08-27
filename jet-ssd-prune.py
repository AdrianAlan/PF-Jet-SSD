import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import yaml

from ssd.net import build_ssd
from utils import *


def is_pooling_layer(layer):
    """
    Checks if layer is a pooling layer.
    """
    return isinstance(layer, nn.AvgPool2d)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Prune Jet Detection Model')
    parser.add_argument('model', type=str,
                        help='Network to prune')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-o', '--out', action=IsValidFile, type=str,
                        help='Path to output config file',
                        default='net-config-last.yml')
    parser.add_argument('-s', '--structure', action=IsValidFile, type=str,
                        help='Path to config file', default='net-config.yml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    ssd_settings = config['ssd_settings']
    ssd_settings['n_classes'] += 1
    input_dimensions = ssd_settings['input_dimensions']

    net_config = yaml.safe_load(open(args.structure))
    channels = net_config['network_channels']

    path = '{}/{}.pth'.format(config['output']['model'], args.model)
    net = build_ssd("cpu", config['ssd_settings'], channels)
    net.load_weights(path)
    net.eval()

    # Copy attention parameters
    state_dict = net.state_dict()
    tmp_attention1 = state_dict['attention1.conv1d.weight']
    tmp_attention2 = state_dict['attention2.conv1d.weight']

    # Define target channels
    # max_channels = [32, None, 64, 128, 128, None, 256, 512, 512, None,
    #                 512, 512, None, 1024, 1024]
    # max_channels = [32, None, 64, 128, 128, None, 230, 435, 435, None,
    #                 435, 435, None, 845, 845]
    # max_channels = [32, None, 64, 128, 128, None, 205, 358, 358, None,
    #                 358, 358, None, 666, 666]
    # max_channels = [32, None, 64, 128, 128, None, 179, 281, 281, None,
    #                 281, 281, None, 487, 487]
    # max_channels = [32, None, 64, 128, 128, None, 153, 204, 204, None,
    #                 204, 204, None, 308, 308]
    max_channels = [32, None, 64, 128, 128, None, 128, 128, 128, None,
                    128, 128, None, 128, 128]

    # Prune the model
    dummy_input = torch.unsqueeze(torch.randn(input_dimensions), 0)
    DG = tp.DependencyGraph()
    DG.build_dependency(net, example_inputs=dummy_input)

    for x, (m, c) in enumerate(zip(net.mobilenet.children(), max_channels)):

        if is_pooling_layer(m):
            continue

        if len(m) == 4:
            _, gamma = list(m.named_parameters())[1]
            gamma = torch.abs(gamma)
            threshold_distr = torch.mean(gamma)-torch.std(gamma)
            threshold_fixed = sorted(gamma)[len(gamma)-min(len(gamma), c)]
            threshold = max(threshold_distr, threshold_fixed)
            pruning_idxs = (gamma < threshold).nonzero().reshape(-1).tolist()
            if len(pruning_idxs):
                pruning_plan = DG.get_pruning_plan(net.mobilenet[x][1],
                                                   tp.prune_batchnorm,
                                                   idxs=pruning_idxs)
                pruning_plan.exec()

        if len(m) == 6:
            _, gamma = list(m.named_parameters())[-3]
            gamma = torch.abs(gamma)
            threshold_distr = torch.mean(gamma)-torch.std(gamma)
            threshold_fixed = sorted(gamma)[len(gamma)-min(len(gamma), c)]
            threshold = max(threshold_distr, threshold_fixed)
            pruning_idxs = (gamma < threshold).nonzero().reshape(-1).tolist()
            if len(pruning_idxs):
                pruning_plan = DG.get_pruning_plan(net.mobilenet[x][-2],
                                                   tp.prune_batchnorm,
                                                   idxs=pruning_idxs)
                pruning_plan.exec()

    # Export new channel config
    ch = []
    for x, child in enumerate(net.mobilenet.children()):
        if not is_pooling_layer(child):
            if len(child) == 4:
                ch.append(child[0].out_channels)
            if len(child) == 6:
                ch.append(child[-3].out_channels)

    with open(args.out, 'w') as f:
        yaml.dump({"network_channels": ch}, f)

    # Restore the attention parameters
    state_dict = net.state_dict()
    del state_dict['attention1.conv1d.weight']
    del state_dict['attention2.conv1d.weight']
    state_dict['attention1.conv1d.weight'] = tmp_attention1
    state_dict['attention2.conv1d.weight'] = tmp_attention2

    # Save the new model
    torch.save(state_dict, path)
