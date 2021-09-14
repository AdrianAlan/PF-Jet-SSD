import argparse
import onnx
import onnxruntime
import torch
import torch.nn as nn
import yaml

from ssd.net import build_ssd
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert PyTorch SSD to ONNX")
    parser.add_argument('model',
                        type=str,
                        help='Input model name')
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='ssd-config.yml')
    parser.add_argument('-s', '--structure',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='net-config.yml')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    net_config = yaml.safe_load(open(args.structure))
    net_channels = net_config['network_channels']

    logger = set_logging('CEVA_SSD',
                         '{}/PF-Jet-SSD-Test.log'.format(
                              config['output']['model']),
                         args.verbose)

    logger.info('Converting {} model to ONNX'.format(args.model))

    ssd_settings = config['ssd_settings']
    input_dimensions = ssd_settings['input_dimensions']
    jet_size = ssd_settings['object_size']
    num_workers = config['evaluation_pref']['workers']
    dataset = config['dataset']['validation'][0]

    ssd_settings['n_classes'] += 1

    base = '{}/{}'.format(config['output']['model'], args.model)
    source_path = '{}.pth'.format(base)
    export_path = '{}-ceva.onnx'.format(base)

    torch.set_default_tensor_type('torch.FloatTensor')

    logger.info('Prepare PyTorch model')
    net = build_ssd(torch.device('cpu'),
                    ssd_settings,
                    net_channels,
                    ceva=True,
                    inference=True,
                    onnx=True)

    net.load_weights(source_path)
    net.eval()

    logger.info('Prepare inputs')
    loader = get_data_loader(dataset,
                             1,
                             num_workers,
                             input_dimensions,
                             jet_size,
                             cpu=True,
                             shuffle=False)

    batch_iterator = iter(loader)
    dummy_input, _ = next(batch_iterator)
    logger.info('Export as ONNX model')
    torch.onnx.export(net,
                      dummy_input,
                      export_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    logger.info('Validating graph')
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    logger.info("Exported model has been successfully tested with ONNXRuntime")
