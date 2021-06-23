import argparse
import numpy as np
import yaml

from utils import IsValidFile, Plotting

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Plot SSD's Inference Time")
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='ssd-config.yml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    latency_pytorch_fp32 = [158.69, 254.08, 472.38, 1009.10, 9352.68, np.nan, np.nan, np.nan, np.nan]
    latency_pytorch_int8 = [40.61, 61.07, 129.62, 269.42, 648.28, 1346.26, 2728.85, 5437.04, 10635.50]
    latency_onnx_runtime_fp32 = [110.59, 234.71, 486.53, 994.54, 2003.04, 3979.58,  7965.95, 17490.42, np.nan]
    latency_onnx_runtime_int8 = [147.63, 303.70, 613.34, 1252.22, 2511.32, 5024.10, 10013.98, 22700.14, np.nan]
    latency_tensorrt_fp32 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    latency_tensorrt_fp16 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    latency_tensorrt_int8 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    throughput_pytorch_fp32 = [6.30, 7.87, 8.47, 7.93, 1.71, np.nan, np.nan, np.nan, np.nan]
    throughput_pytorch_int8 = [24.62, 32.75, 30.86, 29.69, 24.68, 23.77, 23.45, 23.54, 24.07]
    throughput_onnx_runtime_fp32 = [9.04, 8.52, 8.22, 8.04, 7.99, 8.04, 8.03, 7.32, np.nan]
    throughput_onnx_runtime_int8 = [6.77, 6.59, 6.52, 6.39, 6.37, 6.37, 6.39, 5.64, np.nan]
    throughput_tensorrt_fp32 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    throughput_tensorrt_fp16 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    throughput_tensorrt_int8 = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    plot = Plotting(save_dir=config['output']['plots'])

    batch_sizes = ['$2^{}$'.format(i) for i in range(9)]

    labels = ['FP32 CPU/PyTorch',
              'INT8 CPU/PyTorch',
              'FP32 CPU/ONNX',
              'INT8 CPU/ONNX',
              'FP32 GPU/TensorRT',
              'FP16 GPU/TensorRT',
              'INT8 GPU/TensorRT']

    plot.draw_barchart(batch_sizes,
                       [latency_pytorch_fp32,
                        latency_pytorch_int8,
                        latency_onnx_runtime_fp32,
                        latency_onnx_runtime_int8,
                        latency_tensorrt_fp32,
                        latency_tensorrt_fp16,
                        latency_tensorrt_int8],
                       labels,
                       ylabel='Latency [ms]',
                       save_name='inference-latency')

    plot.draw_barchart(batch_sizes,
                       [throughput_pytorch_fp32,
                        throughput_pytorch_int8,
                        throughput_onnx_runtime_fp32,
                        throughput_onnx_runtime_int8,
                        throughput_tensorrt_fp32,
                        throughput_tensorrt_fp16,
                        throughput_tensorrt_int8],
                       labels,
                       ylabel='Throughput [eps]',
                       save_name='inference-throughput')
