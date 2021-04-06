import argparse
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import yaml

from ssd.net import build_ssd
from utils import *


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert PyTorch SSD to ONNX")
    parser.add_argument('model', type=str, help='Input model name')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    base_name = '{}/{}'.format(config['output']['model'], args.model)
    source_path = '{}.pth'.format(base_name)
    export_path = '{}.onnx'.format(base_name)
    log_path = '{}.log'.format(base_name)

    logger = set_logging('Test_SSD', log_path, args.verbose)
    logger.info('Converting PyTorch SSD model to ONNX')

    logger.info('Prepare PyTorch model')
    ssd_settings = config['ssd_settings']
    ssd_settings['n_classes'] += 1
    net = build_ssd('cpu', ssd_settings, inference=True, onnx=True)
    net.load_weights(source_path)
    net.eval()

    logger.info('Prepare inputs')
    in_dim = ssd_settings['input_dimensions']
    jet_size = ssd_settings['object_size']
    workers = config['evaluation_pref']['workers']
    loader = get_data_loader(config['dataset']['validation'][0], int(1),
                             workers, in_dim, jet_size, shuffle=False)
    batch_iterator = iter(loader)
    dummy_input, _ = next(batch_iterator)
    dummy_input = dummy_input.cpu()

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

    logger.info('Matching outputs')
    ort_session = onnxruntime.InferenceSession(export_path)
    # Compute PyTorch output prediction
    torch_outs = list(map(to_numpy, list(net(dummy_input))))
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # Compare ONNX Runtime and PyTorch results
    for i, task in enumerate(['Localization', 'Classification', 'Regression']):
        np.testing.assert_allclose(torch_outs[i],
                                   ort_outs[i],
                                   rtol=1e-03,
                                   atol=1e-05)
        logger.info('{} task: OK'.format(task))

    logger.info("Exported model has been successfully tested with ONNXRuntime")
