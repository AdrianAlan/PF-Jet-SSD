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

    latency_pytorch_fp32 = [134.64, 205.19, 339.34, 659.04, 1456.34, 3950.28, 7948.51, 13540.88, 31960.05]
    latency_pytorch_int8 = [232.85, 421.56, 834.81, 1545.68, 2980.03, 6140.19, 11674.74, 23024.93, 44790.11]
    latency_onnx_runtime_fp32 = [56.48, 144.02, 259.35, 542.03, 996.81, 2180.52, 4220.26, 8598.74, 18819.14]
    latency_onnx_runtime_int8 = [81.23, 160.64, 338.66, 747.45, 1442.26, 2840.83, 5901.70, 11678.05, 26420.73]
    latency_tensorrt_fp32 = [1.81, 3.17, 5.46, 10.24, 19.60, 38.11, 75.49, 150.43, 300.03]
    latency_tensorrt_fp16 = [1.13, 1.94, 3.48, 6.63, 12.85, 26.61, 53.78, 102.04, 199.37]
    latency_tensorrt_int8 = [1.22, 1.70, 2.87, 5.17, 10.49, 21.04, 40.31, 74.29, 140.44]

    throughput_pytorch_fp32 = [7.43, 9.75, 11.79, 12.14, 10.99, 8.10, 8.05, 9.45, 8.01]
    throughput_pytorch_int8 = [4.29, 4.74, 4.79, 5.18, 5.37, 5.21, 5.48, 5.56, 5.72]
    throughput_onnx_runtime_fp32 = [17.70, 13.89, 15.42, 14.76, 16.05, 14.68, 15.16, 14.89, 13.60]
    throughput_onnx_runtime_int8 = [12.31, 12.45, 11.81, 10.70, 11.09, 11.26, 10.84, 10.96, 9.69]
    throughput_tensorrt_fp32 = [552.18, 630.20, 732.76, 781.24, 816.18, 839.60, 847.80, 850.91, 853.26]
    throughput_tensorrt_fp16 = [887.24, 1033.03, 1150.09, 1206.35, 1244.77, 1202.73, 1190.02, 1254.36, 1284.07]
    throughput_tensorrt_int8 = [821.50, 1173.09, 1394.56, 1548.36, 1524.91, 1521.23, 1587.66, 1722.87, 1822.88]

    plot = Plotting(save_dir=config['output']['plots'])

    batch_sizes = [r'$2^{}$'.format(i) for i in range(9)]

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
                       [throughput_pytorch_fp32,
                        throughput_pytorch_int8,
                        throughput_onnx_runtime_fp32,
                        throughput_onnx_runtime_int8,
                        throughput_tensorrt_fp32,
                        throughput_tensorrt_fp16,
                        throughput_tensorrt_int8],
                       labels,
                       ylabel=['Latency [ms]',
                               'Throughput [eps]'],
                       save_name='Inference-Results')
