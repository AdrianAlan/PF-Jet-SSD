import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import warnings

from ssd.net import build_ssd
from tqdm import tqdm
from utils import *

warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*'
)


def execute(model,
            dataset,
            im_size,
            obj_size,
            conf_threshold=10**-6,
            batch_size=50,
            max_distance=.1,
            num_classes=3,
            epsilon=10**-6,
            text='Evaluating Network',
            verbose=False):

    results = [torch.empty(0, 6).cpu() for _ in range(num_classes)]
    deltas = torch.empty((0, 5))

    if verbose:
        progress_bar = tqdm(total=len(dataset), desc=text)

    for X, y, _, scalers in dataset:

        y_pred = model(X).data
        for i in range(batch_size):
            all_detections = torch.empty((0, 6))
            detections, targets = y_pred[i], y[i]
            scaler = scalers[i]

            for cid, dts in enumerate(detections):

                # Filter detections above given threshold
                dts = dts[dts[:, 0] > conf_threshold]

                if dts.size(0) == 0:
                    continue

                dx = (dts[:, 1] + dts[:, 3]).unsqueeze(1)/2
                dy = (dts[:, 2] + dts[:, 4]).unsqueeze(1)/2
                scores = dts[:, 0].unsqueeze(1)
                regres = dts[:, 5].unsqueeze(1) * scaler
                labels = cid*torch.ones(len(scores)).unsqueeze(1)
                truths = torch.zeros(len(scores)).unsqueeze(1)

                # Format: [x, y, label, score, truth, pt]
                dts = torch.cat((dx, dy, labels, scores, truths, regres), 1)
                all_detections = torch.cat((all_detections, dts))

            # Sort by confidence
            all_detections = all_detections[(-all_detections[:, 3]).argsort()]

            for t in targets:
                detected = False
                tx = (t[0]+t[2])/2
                ty = (t[1]+t[3])/2
                tp = t[5] * scaler

                for x, d in enumerate(all_detections):
                    delta_eta = (tx-d[0])
                    delta_phi = torch.min((ty-d[1]) % 1, (d[1]-ty) % 1)
                    distance = torch.sqrt(delta_eta**2+delta_phi**2)

                    if distance > max_distance:
                        continue

                    if d[2] == t[4]:
                        detected = True

                        # Angular resolution and regression data
                        deta = np.radians(1)*im_size[0]*delta_eta
                        dphi = np.radians(1)*im_size[1]*delta_phi
                        dpt = 1 - d[5] / (tp + epsilon)
                        dts = torch.Tensor([t[4], tp, deta, dphi, dpt])
                        deltas = torch.cat((deltas, dts.unsqueeze(0)))

                        all_detections[x][0] = tx
                        all_detections[x][1] = ty
                        all_detections[x][4] = 1
                        all_detections[x][5] = tp
                        break

                if not detected:
                    fn = torch.Tensor([tx, ty, t[4], 0, 1, tp])
                    fn = fn.unsqueeze(0)
                    all_detections = torch.cat((all_detections, fn))

            for c in range(num_classes):
                dets = all_detections[all_detections[:, 2] == (c + 1)].cpu()
                results[c] = torch.cat((results[c], dets)).cpu()

        if args.verbose:
            progress_bar.update(1)

    if args.verbose:
        progress_bar.close()

    return results, deltas.cpu()


def execute_baseline(dataset,
                     im_size,
                     obj_size,
                     batch_size=50,
                     max_distance=.1,
                     num_classes=3,
                     text='Evaluating Baseline',
                     epsilon=10**-6,
                     verbose=False):

    results = [torch.empty(0, 6).cpu() for _ in range(num_classes)]
    deltas = torch.empty((0, 5))

    if verbose:
        progress_bar = tqdm(total=len(dataset), desc=text)

    for _, y, baseline, scalers in dataset:

        for i in range(batch_size):
            targets, scaler = y[i], scalers[i]

            all_baselines = baseline[i]
            # Sort by confidence
            all_baselines = all_baselines[(-all_baselines[:, 3]).argsort()]

            for t in targets:
                detected = False
                tx = (t[0]+t[2])/2
                ty = (t[1]+t[3])/2
                tp = t[5] * scaler

                for x, b in enumerate(all_baselines):
                    delta_eta = (tx-b[0])
                    delta_phi = torch.min((ty-b[1]) % 1, (b[1]-ty) % 1)
                    distance = torch.sqrt(delta_eta**2+delta_phi**2)

                    if distance > max_distance:
                        continue

                    if b[2] == t[4]:
                        detected = True

                        # Angular resolution and regression data
                        deta = np.radians(1)*im_size[0]*delta_eta
                        dphi = np.radians(1)*im_size[1]*delta_phi
                        dpt = 1 - b[5] / (tp + epsilon)
                        dts = torch.Tensor([t[4], tp, deta, dphi, dpt])
                        deltas = torch.cat((deltas, dts.unsqueeze(0)))

                        all_baselines[x][0] = tx
                        all_baselines[x][1] = ty
                        all_baselines[x][4] = 1
                        all_baselines[x][5] = tp
                        break

                if not detected:
                    fn = torch.Tensor([tx, ty, t[4], 0, 1, tp])
                    fn = fn.unsqueeze(0)
                    all_baselines = torch.cat((all_baselines, fn))

            for c in range(num_classes):
                dets = all_baselines[all_baselines[:, 2] == (c + 1)].cpu()
                results[c] = torch.cat((results[c], dets))

        if args.verbose:
            progress_bar.update(1)

    if args.verbose:
        progress_bar.close()

    return results, deltas.cpu()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate Jet Detection Model')
    parser.add_argument('fpn', type=str,
                        help='Full Precision Network model name')
    parser.add_argument('twn', type=str,
                        help='Ternary Weight Network model name')
    parser.add_argument('int8', type=str,
                        help='int8 Network model name')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-s', '--structure', action=IsValidFile, type=str,
                        help='Path to config file', default='net-config.yml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    net_config = yaml.safe_load(open(args.structure))

    dataset = config['dataset']['test'][0]
    evaluation_pref = config['evaluation_pref']
    ssd_settings = config['ssd_settings']
    net_channels = net_config['network_channels']
    batch_size = evaluation_pref['batch_size']
    jet_names = evaluation_pref['names_classes']
    num_workers = evaluation_pref['workers']
    conf_threshold = ssd_settings['confidence_threshold']
    input_dimensions = ssd_settings['input_dimensions']
    jet_size = ssd_settings['object_size']
    num_classes = ssd_settings['n_classes']
    ssd_settings['n_classes'] += 1
    max_distance = ssd_settings['max_distance']

    plotting_results = []
    plotting_deltas = []

    logger = set_logging('Test_SSD',
                         '{}/PF-Jet-SSD-Test.log'.format(
                             config['output']['model']),
                         args.verbose)
    logger.info('Testing baseline')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    loader = get_data_loader(dataset,
                             batch_size,
                             num_workers,
                             input_dimensions,
                             jet_size,
                             cpu=False,
                             return_baseline=True,
                             return_pt=True,
                             shuffle=False)

    base_results, base_deltas = execute_baseline(loader,
                                                 input_dimensions[1:],
                                                 jet_size,
                                                 batch_size,
                                                 max_distance,
                                                 num_classes,
                                                 text='Evaluating Baseline',
                                                 verbose=args.verbose)

    for i, name in enumerate([args.fpn, args.twn, args.int8]):
        logger.info('Testing {0}'.format(name))
        path = '{}/{}.pth'.format(config['output']['model'], name)

        if i == 2:
            torch.set_default_tensor_type('torch.FloatTensor')
            net = build_ssd(torch.device('cpu'),
                            ssd_settings,
                            net_channels,
                            inference=True,
                            int8=True)
            net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(net, inplace=True)
            net.load_weights(path)
            net = net.cpu()
            torch.quantization.convert(net.eval(), inplace=True)
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            net = build_ssd(0, ssd_settings, net_channels, inference=True)
            net.load_weights(path)
            cudnn.benchmark = True
            net = net.cuda()
            net.eval()

        if i == 0:
            dummy_input = torch.unsqueeze(torch.randn(input_dimensions), 0)
            mac = GetResources(net, dummy_input).profile() / 1e9
            logger.info('Total OPS: {0:.3f}G'.format(mac))
            logger.info('Total network parameters: {0}'.format(
                sum(p.numel() for p in net.parameters() if p.requires_grad)))

        loader = get_data_loader(dataset,
                                 batch_size,
                                 num_workers,
                                 input_dimensions,
                                 jet_size,
                                 cpu=(i == 2),
                                 return_baseline=True,
                                 return_pt=True,
                                 shuffle=False)

        with torch.no_grad():
            results, deltas = execute(net,
                                      loader,
                                      input_dimensions[1:],
                                      jet_size,
                                      conf_threshold=conf_threshold,
                                      batch_size=batch_size,
                                      max_distance=max_distance,
                                      num_classes=num_classes,
                                      text='Evaluating {}'.format(name),
                                      verbose=args.verbose)

        plotting_results.append(results)
        plotting_deltas.append(deltas)

    plot = Plotting(save_dir=config['output']['plots'])

    plot.draw_precision_recall(base_results,
                               plotting_results[0],
                               plotting_results[1],
                               plotting_results[2],
                               jet_names)

    plot.draw_precision_details(base_deltas,
                                plotting_results[0],
                                plotting_results[1],
                                plotting_results[2],
                                jet_names)

    plot.draw_loc_delta(base_deltas,
                        plotting_deltas[0],
                        plotting_deltas[1],
                        plotting_deltas[2],
                        jet_names)
