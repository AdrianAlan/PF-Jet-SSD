import argparse
import yaml

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Plot SSD's Inference Time")
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    latency32 = [2.8129, 5.3436, 10.1207, 19.6781, 39.0229, 78.3424, 158.4441]
    latency16 = [1.2757, 2.4261, 4.6120, 9.0185, 17.7228, 35.5387, 71.3110]
    throughput32 = [355.49, 374.28, 395.23, 406.54, 410.02, 408.46, 403.93]
    throughput16 = [783.84, 824.34, 867.30, 887.06, 902.79, 900.42, 897.48]
    plot = Plotting(save_dir=config['output']['plots'])
    plot.draw_barchart([1, 2, 4, 8, 16, 32, 64],
                       [latency32, latency16],
                       ['FP32 GPU', 'FP16/TWN GPU (Not optimized)'],
                       ylabel='Latency [ms]',
                       save_name='inference-latency')
    plot.draw_barchart(['$2^0$', '$2^1$', '$2^2$', '$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$', '$2^8$'],
                       [throughput32, throughput16],
                       ['FP32 GPU', 'FP16/TWN GPU'],
                       ylabel='Throughput [eps]',
                       save_name='inference-throughput')
