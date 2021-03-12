import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml

from ssd.net import build_ssd
from tqdm import tqdm
from utils import *


def execute(model, dataset, im_size, obj_size, conf_threshold=10**-6,
            batch_size=50, max_distance=.1, num_classes=3, epsilon=10**-6,
            verbose=False):

    results = [torch.empty(0, 3) for _ in range(num_classes)]
    results_baseline = [torch.empty(0, 3) for _ in range(num_classes)]
    deltas, deltas_baseline = torch.empty((0, 5)), torch.empty((0, 5))

    if args.verbose:
        progress_bar = tqdm(total=len(dataset), desc='Evaluating events')

    for X, y, base in dataset:

        pred = model(X).data
        for idx in range(batch_size):
            detections, targets, all_baselines = pred[idx], y[idx], base[idx]
            all_detections = torch.empty((0, 8))

            for cid, dts in enumerate(detections):

                # Filter detections above given threshold
                dts = dts[dts[:, 0] > conf_threshold]

                if dts.size(0) == 0:
                    continue

                bboxes = dts[:, 1:5]
                scores = dts[:, 0].unsqueeze(1)
                regres = dts[:, 5].unsqueeze(1)
                labels = cid*torch.ones(len(scores)).unsqueeze(1)
                truths = torch.zeros(len(scores)).unsqueeze(1)

                # Format: [xmin, ymin, xmax, ymax, label, score, truth, pt]
                dts = torch.cat((bboxes, labels, scores, truths, regres), 1)
                all_detections = torch.cat((all_detections, dts))

            # Sort by confidence
            all_detections = all_detections[(-all_detections[:, 5]).argsort()]
            all_baselines = all_baselines[(all_baselines[:, 5]).argsort()]

            for t in targets:
                detected, dbaseline = False, False
                tx = (t[0]+t[2])/2
                ty = (t[1]+t[3])/2

                for x, d in enumerate(all_detections):
                    dx = (d[0]+d[2])/2
                    dy = (d[1]+d[3])/2
                    delta_eta = (tx-dx)
                    delta_phi = torch.min((ty-dy) % 1, (dy-ty) % 1)
                    distance = torch.sqrt(delta_eta**2+delta_phi**2)

                    if distance > max_distance:
                        continue

                    if d[4] == t[4]:
                        detected = True
                        all_detections[x][6] = 1

                        # Angular resolution and regression data
                        deta = np.radians(1)*im_size[0]*delta_eta
                        dphi = np.radians(1)*im_size[1]*delta_phi
                        dpt = 1 - d[7] / (t[5] + epsilon)
                        dts = torch.Tensor([t[4], t[5], deta, dphi, dpt])
                        deltas = torch.cat((deltas, dts.unsqueeze(0)))

                        all_detections[x][7] = t[5]
                        break

                for x, b in enumerate(all_baselines):
                    bx = (b[0]+b[2])/2
                    by = (b[1]+b[3])/2
                    delta_eta = (tx-bx)
                    delta_phi = torch.min((ty-by) % 1, (dy-by) % 1)
                    distance = torch.sqrt(delta_eta**2+delta_phi**2)

                    if distance > max_distance:
                        continue

                    if b[4] == t[4]:
                        dbaseline = True
                        all_baselines[x][6] = 1

                        # Angular resolution and regression data
                        deta = np.radians(1)*im_size[0]*delta_eta
                        dphi = np.radians(1)*im_size[1]*delta_phi
                        dpt = 1 - b[7] / (t[5] + epsilon)
                        dts = torch.Tensor([b[4], b[5], deta, dphi, dpt])
                        deltas_baseline = torch.cat((deltas_baseline,
                                                     dts.unsqueeze(0)))
                        all_baselines[x][7] = t[5]
                        break

                if not detected:
                    fn = torch.cat((t[:5], torch.Tensor([0, 1, t[5]])))
                    fn = fn.unsqueeze(0)
                    all_detections = torch.cat((all_detections, fn))

                if not dbaseline:
                    fn = torch.cat((t[:5], torch.Tensor([0, 1, t[5]])))
                    fn = fn.unsqueeze(0)
                    all_baselines = torch.cat((all_baselines, fn))

            for c in range(num_classes):
                cls_dets = all_detections[all_detections[:, 4] == (c + 1)]
                results[c] = torch.cat((results[c], cls_dets[:, [5, 6, 7]]))

                cls_dets = all_baselines[all_baselines[:, 4] == (c + 1)]
                results_baseline[c] = torch.cat((results_baseline[c],
                                                 cls_dets[:, [5, 6, 7]]))
        if args.verbose:
            progress_bar.update(1)

    if args.verbose:
        progress_bar.close()

    return results, results_baseline, deltas, deltas_baseline


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate Jet Detection Model')
    parser.add_argument('fpn', type=str,
                        help='Full Precision Network model name')
    parser.add_argument('twn', type=str,
                        help='Ternary Weight Network model name')
    parser.add_argument('-c', '--config', action=IsValidFile, type=str,
                        help='Path to config file', default='ssd-config.yml')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    if not torch.cuda.is_available():
        pass
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    evaluation_pref = config['evaluation_pref']
    ssd_settings = config['ssd_settings']
    bs = evaluation_pref['batch_size']
    jet_names = evaluation_pref['names_classes']
    workers = evaluation_pref['workers']
    ct = ssd_settings['confidence_threshold']
    in_dim = ssd_settings['input_dimensions']
    jet_size = ssd_settings['object_size']
    num_classes = ssd_settings['n_classes']
    ssd_settings['n_classes'] += 1
    md = ssd_settings['max_distance']

    plotting_results = []
    plotting_deltas = []

    for i, name in enumerate([args.fpn, args.twn]):
        base = '{}/{}'.format(config['output']['model'], name)
        source_path = '{}.pth'.format(base)

        logger = set_logging('Test_SSD', '{}.log'.format(base), args.verbose)
        logger.info('Testing {0} model'.format(source_path))

        net = build_ssd(0, ssd_settings, inference=True)
        net.load_weights(source_path)
        cudnn.benchmark = True
        net = net.cuda()
        net.eval()
        qbits = 8 if i else None
        loader = get_data_loader(config['dataset']['test'][0], bs, workers,
                                 in_dim, jet_size, return_baseline=True,
                                 return_pt=True, qbits=qbits, shuffle=False)

        with torch.no_grad():
            results, results_baseline, deltas, deltas_baseline = execute(
                                                      net,
                                                      loader,
                                                      in_dim[1:],
                                                      jet_size,
                                                      conf_threshold=ct,
                                                      batch_size=bs,
                                                      max_distance=md,
                                                      num_classes=num_classes,
                                                      verbose=args.verbose)

        plotting_results.append(results)
        plotting_deltas.append(deltas)

        dummy_input = torch.unsqueeze(torch.randn(in_dim), 0)
        mac = GetResources(net, dummy_input=dummy_input).profile() / 1e9
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.debug('Total OPS: {0:.3f}G'.format(mac))
        logger.debug('Total network parameters: {0}'.format(params))

    plot = Plotting(save_dir=config['output']['plots'])
    plot.draw_precision_recall(plotting_results[0],
                               plotting_results[1],
                               results_baseline,
                               jet_names)
    plot.draw_precision_in_pt(plotting_results[0],
                              plotting_results[1],
                              results_baseline,
                              jet_names)
    plot.draw_loc_delta(plotting_deltas[0],
                        plotting_deltas[1],
                        deltas_baseline,
                        jet_names)
