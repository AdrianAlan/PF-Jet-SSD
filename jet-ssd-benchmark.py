import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
import time
import warnings
import yaml

from numpy.testing import assert_almost_equal as is_equal
from ssd.net import build_ssd
from utils import *
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

warnings.filterwarnings(
     action='ignore',
     category=UserWarning,
     module=r'.*'
)


class ImageBatchStream():
    def __init__(self, batch_size, data_loader):
        self.batch = 0
        self.batch_size = batch_size
        self.max_batches = 10
        self.calibration_data = np.zeros(
            (self.max_batches, batch_size, 3, 340, 360),
            dtype=np.float32)

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            images = []
            for batch_index, (image, _) in enumerate(data_loader):
                if batch_index < self.max_batches:
                    images.append(to_numpy(image))
                else:
                    break
            for i in range(self.max_batches):
                self.calibration_data[i] = images[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data,
                                        dtype=np.float32)
        else:
            return np.array([])


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, loader):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.loader = loader
        self.loader.reset()
        self.d_input = cuda.mem_alloc(self.loader.calibration_data.nbytes)

    def get_batch_size(self):
        return self.loader.batch_size

    def get_batch(self, bindings):
        batch = self.loader.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, size):
        return None


def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()


def create_model_for_provider(model_path: str,
                              provider: str) -> InferenceSession:
    all_providers = get_all_providers()
    assert provider in all_providers, f'{provider} not in {all_providers}'
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return InferenceSession(model_path, so, providers=[provider])


def run_pytorch_benchmark(model,
                          data_loader,
                          batch_size,
                          samples):
    images = []
    for batch_index, (image, _) in enumerate(data_loader):
        if batch_index < samples:
            image = image.cpu()
            images.append(image)
        else:
            break

    logger.info('Taking measurements')
    measurements = 0
    for image in images:
        t_start = time.time()
        _ = model(image)
        elapsed_time = time.time() - t_start
        measurements += elapsed_time * 1e3
    latency = measurements / samples
    throughput = 1e3 * batch_size / latency
    return latency, throughput


def run_onnx_benchmark(model,
                       data_loader,
                       batch_size,
                       samples):

    logger.info('Loading ONNX model')
    cpu_model = create_model_for_provider(model, "CPUExecutionProvider")

    images = []
    for batch_index, (image, _) in enumerate(data_loader):
        if batch_index < samples:
            image = to_numpy(image)
            images.append(image)
        else:
            break

    logger.info('Taking measurements')
    measurements = 0
    for image in images:
        t_start = time.time()
        _ = cpu_model.run(None, {'input': image})
        elapsed_time = time.time() - t_start
        measurements += elapsed_time * 1e3
    latency = measurements / samples
    throughput = 1e3 * batch_size / latency
    return latency, throughput


def run_tensorrt_benchmark(net,
                           onnx_model,
                           data_loader,
                           batch_size,
                           samples,
                           input_dimensions,
                           suppress=False,
                           fp16=False,
                           int8=False):
    images = []
    for batch_index, (image, _) in enumerate(data_loader):
        if batch_index < samples:
            images.append(image)
        else:
            break

    lgr = trt.Logger(trt.Logger.INFO)
    net_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(lgr) as builder, \
        builder.create_network(net_flag) as network, \
        trt.OnnxParser(network, lgr) as parser, \
        builder.create_builder_config() as cfg:

        builder.max_batch_size = batch_size
        builder.max_workspace_size = 1 << 30

        logger.info('Parsing ONNX file')
        with open(source_path_onnx, 'rb') as model:
            if not parser.parse(model.read()):
                raise RuntimeError('Parsing failed! Error: {:}'.format(
                    parser.get_error(0).desc()))

        size = (batch_size, *input_dimensions)
        profile = builder.create_optimization_profile()
        profile.set_shape("input", size, size, size)
        cfg.add_optimization_profile(profile)

        if fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
            cfg.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if int8:
            batchstream = ImageBatchStream(batch_size, data_loader)
            int8_calibrator = PythonEntropyCalibrator(batchstream)
            cfg.int8_calibrator = int8_calibrator
            cfg.set_flag(trt.BuilderFlag.INT8)
            cfg.set_flag(trt.BuilderFlag.STRICT_TYPES)

        logger.info('Building TensorRT engine')
        engine = builder.build_engine(network, cfg)

        host_in, cuda_in, host_out, cuda_out, bindings = [], [], [], [], []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(size, dtype=np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))

            if engine.binding_is_input(binding):
                host_in.append(host_mem)
                cuda_in.append(cuda_mem)
            else:
                host_out.append(host_mem)
                cuda_out.append(cuda_mem)
                context = engine.create_execution_context()

        logger.info('Taking measurements')
        measurements = 0
        warmup = True
        for image in images:

            np.copyto(host_in[0], to_numpy(image).ravel())

            if warmup:
                logger.info('GPU warm-up')
                for _ in range(10):
                    cuda.memcpy_htod_async(cuda_in[0], host_in[0], stream)
                    context.execute_async(batch_size=batch_size,
                                          bindings=bindings,
                                          stream_handle=stream.handle)
                warmup = False

            t_start = time.time()
            cuda.memcpy_htod_async(cuda_in[0], host_in[0], stream)
            context.execute_async(batch_size=batch_size,
                                  bindings=bindings,
                                  stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_out[2], cuda_out[2], stream)
            cuda.memcpy_dtoh_async(host_out[1], cuda_out[1], stream)
            cuda.memcpy_dtoh_async(host_out[0], cuda_out[0], stream)
            stream.synchronize()
            elapsed_time = time.time() - t_start
            measurements += elapsed_time * 1e3

            if not suppress:
                desired = list(map(to_numpy, list(net(image))))
                for i, task in enumerate(['loc', 'cls', 'reg']):
                    is_equal(host_out[i].reshape(desired[i].shape),
                             desired[i],
                             decimal=3)

    latency = measurements / samples
    throughput = 1e3 * batch_size / latency

    return latency, throughput


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Measure SSD's Inference Time")
    parser.add_argument('model',
                        type=str,
                        help='Input model name')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        help='Test batch size',
                        default=1)
    parser.add_argument('-8',  '--int8',
                        action='store_true',
                        help='Run network in int8')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Run network in FP16')
    parser.add_argument('--onnx',
                        action='store_true',
                        help='Run network in ONNX')
    parser.add_argument('--trt',
                        action='store_true',
                        help='Run network in TensorRT')
    parser.add_argument('-s', '--suppress',
                        action='store_true',
                        help='Suppress checks')
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='ssd-config.yml')
    parser.add_argument('-n', '--structure',
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

    ssd_settings = config['ssd_settings']
    net_channels = net_config['network_channels']
    ssd_settings['n_classes'] += 1
    input_dimensions = ssd_settings['input_dimensions']
    jet_size = ssd_settings['object_size']
    num_workers = config['evaluation_pref']['workers']
    samples = config['inference_pref']['samples']

    logger = set_logging('Benchmark_SSD',
                         '{}/PF-Jet-SSD-Benchmark.log'.format(
                              config['output']['model']),
                         args.verbose)
    logger.info('Benchmarking {} model'.format(args.model))

    base = '{}/{}'.format(config['output']['model'], args.model)
    source_path_torch = '{}.pth'.format(base)

    data_loader = get_data_loader(config['dataset']['validation'][0],
                                  args.batch_size,
                                  num_workers,
                                  input_dimensions,
                                  jet_size,
                                  0,
                                  cpu=args.trt,
                                  shuffle=False)

    latency, throughput = 'N/A', 'N/A'

    if args.trt:
        source_path_onnx = '{}.onnx'.format(base)
        net = build_ssd(0,
                        ssd_settings,
                        net_channels,
                        inference=True,
                        onnx=True)
        net.load_weights(source_path_torch)
        net.eval()
        latency, throughput = run_tensorrt_benchmark(net,
                                                     source_path_onnx,
                                                     data_loader,
                                                     args.batch_size,
                                                     samples,
                                                     input_dimensions,
                                                     suppress=args.suppress,
                                                     fp16=args.fp16,
                                                     int8=args.int8)

    if args.onnx:
        if args.int8:
            source_path_onnx = '{}-int8.onnx'.format(base)
        elif args.fp16:
            raise NotImplementedError('ONNX FP16 on CPU not supported')
        else:
            source_path_onnx = '{}.onnx'.format(base)
        # Checks were already performed in onnx export
        latency, throughput = run_onnx_benchmark(source_path_onnx,
                                                 data_loader,
                                                 args.batch_size,
                                                 samples)

    if not args.onnx and not args.trt:
        torch.set_default_tensor_type('torch.FloatTensor')
        net = build_ssd(torch.device('cpu'),
                        ssd_settings,
                        net_channels,
                        inference=True,
                        int8=args.int8,
                        onnx=True)
        if args.int8:
            net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(net, inplace=True)
            net.load_weights(source_path_torch)
            net = net.cpu()
            torch.quantization.convert(net.eval(), inplace=True)
        elif args.fp16:
            raise NotImplementedError('PyTorch FP16 on CPU not supported')
        else:
            net.load_weights(source_path_torch)
            net.eval()
            net = net.cpu()
        latency, throughput = run_pytorch_benchmark(net,
                                                    data_loader,
                                                    args.batch_size,
                                                    samples)

    logger.info('Batch size {0}'.format(args.batch_size))
    logger.info('Latency: {0:.2f} ms'.format(latency))
    logger.info('Throughput: {0:.2f} eps'.format(throughput))
