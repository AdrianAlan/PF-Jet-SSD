import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import yaml

from sklearn.metrics import average_precision_score, precision_recall_curve
from ssd.net import build_ssd
from timeit import default_timer as timer
from tqdm import tqdm
from utils import *


def execute(model, dataset, im_size, conf_threshold=0., batch_size=50,
            overlap_threshold=.1, num_classes=3, epsilon=10**-6, top_k=200,
            verbose=False):

    results = [torch.empty(0, 2) for _ in range(num_classes)]
    deltas = torch.empty((0, 5))
    inf_time = torch.empty(0)

    if args.verbose:
        progress_bar = tqdm(total=len(dataset), desc='Evaluating events')

    for X, y in dataset:

        t_start = timer()
        pred = model(X.cuda()).data
        t_end = timer()
        inf_time = torch.cat((inf_time, torch.Tensor([t_end-t_start])))

        for idx in range(batch_size):
            detections, targets = pred[idx].cuda(), y[idx].cuda()
            targets[:, [0, 2]] *= im_size[0]
            targets[:, [1, 3]] *= im_size[1]
            all_detections = torch.empty((0, 8))

            for class_id, detections in enumerate(detections[1:]):

                # Filter detections above given threshold
                detections = detections[detections[:, 0] > conf_threshold]

                if detections.size(0) == 0:
                    continue

                bboxes = detections[:, 1:5]
                bboxes[:, [0, 2]] *= im_size[0]
                bboxes[:, [1, 3]] *= im_size[1]

                scores = detections[:, 0].unsqueeze(1)
                regres = detections[:, -1].unsqueeze(1)
                labels = (class_id)*torch.ones(len(scores)).unsqueeze(1)
                gt = torch.zeros(len(scores)).unsqueeze(1)

                # Format: [xmin, ymin, xmax, ymax, label, score, gt, m]
                detections = torch.cat((bboxes, labels, scores, gt, regres), 1)
                all_detections = torch.cat((all_detections, detections))

            # Sort by confidence
            all_detections = all_detections[(-all_detections[:, 5]).argsort()]

            # Select top k predictions
            all_detections = all_detections[:top_k]

            for t in targets:
                detected = False

                for x, d in enumerate(all_detections):
                    xmin = torch.max(t[0], d[0])
                    ymin = torch.max(t[1], d[1])
                    xmax = torch.min(t[2], d[2])
                    ymax = torch.min(t[3], d[3])

                    w = torch.max(xmax - xmin, torch.tensor(0.))
                    h = torch.max(ymax - ymin, torch.tensor(0.))
                    intersection = w * h

                    union = ((d[2] - d[0]) * (d[3] - d[1]) +
                             (t[2] - t[0]) * (t[3] - t[1]) - intersection)

                    overlap = intersection / (union + epsilon)

                    if overlap < overlap_threshold:
                        continue

                    if d[4] == t[4]:
                        detected = True
                        all_detections[x][6] = 1

                        # Divide by 115 to get correct resolution
                        d_eta = ((t[0]+t[2])/2 - (d[0]+d[2])/2)/115
                        d_phi = ((t[1]+t[3])/2 - (d[1]+d[3])/2)/115
                        d_mass = (t[5] - d[7]) / (t[5] + epsilon)
                        deltas = torch.cat((
                            deltas,
                            torch.Tensor([[t[4], t[6], d_eta, d_phi, d_mass]])
                        ))
                        break

                if not detected:
                    fn = torch.cat((t[:5], torch.Tensor([0, 1, 1])))
                    fn = fn.unsqueeze(0)
                    all_detections = torch.cat((all_detections, fn))

            for c in range(num_classes):
                cls_dets = all_detections[all_detections[:, 4] == c]
                results[c] = torch.cat((results[c], cls_dets[:, [6, 5]]))

        if args.verbose:
            progress_bar.update(1)

    if args.verbose:
        progress_bar.close()

    it = inf_time.mean()*1000/batch_size

    ret = []
    for c in range(num_classes):
        truth = results[c][:, 0].cpu().numpy()
        score = results[c][:, 1].cpu().numpy()
        p, r, _ = precision_recall_curve(truth, score)
        ap = average_precision_score(truth, score)
        ret.append((r, p, c, ap))

    deltas = torch.abs(deltas)

    return it.cpu().numpy(), ret, deltas.cpu().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate Jet Detection Model')
    parser.add_argument('fpn', type=str,
                        help='Full Precision Network model name')
    parser.add_argument('twn', type=str,
                        help='Ternary Weight Network model name')
    parser.add_argument('config', action=IsValidFile, type=str,
                        help='Path to config file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

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
    ot = ssd_settings['overlap_threshold']
    top_k = ssd_settings['top_k']

    plotting_results = []
    plotting_deltas = []

    loader = get_data_loader(config['dataset']['test'], bs, workers,
                             in_dim, jet_size, return_pt=True, shuffle=False)

    for name in [args.fpn, args.twn]:
        base = '{}/{}'.format(config['output']['model'], name)
        source_path = '{}.pth'.format(base)

        logger = set_logging('Test_SSD', '{}.log'.format(base), args.verbose)
        logger.info('Testing {0} model'.format(source_path))

        net = build_ssd(ssd_settings, inference=True)
        net.load_weights(source_path)
        net.eval()
        net = net.cuda()
        cudnn.benchmark = True

        with torch.no_grad():
            it, res, delta = execute(net, loader, batch_size=bs,
                                     conf_threshold=ct, im_size=in_dim[1:],
                                     num_classes=num_classes,
                                     overlap_threshold=ot, top_k=top_k,
                                     verbose=args.verbose)
        logger.debug('Average inference time: {0:.3f} ms'.format(it))
        for _, _, c, ap in res:
            logger.debug('AP for {0} jets: {1:.3f}'.format(jet_names[c], ap))

        plotting_results.append(res)
        plotting_deltas.append(delta)

        dummy_input = torch.unsqueeze(torch.randn(in_dim), 0)
        mac = GetResources(net, dummy_input=dummy_input).profile() / 1e9
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.debug('Total OPS: {0:.3f}G'.format(mac))
        logger.debug('Total network parameters: {0}'.format(params))

    plot = Plotting(save_dir=config['output']['plots'])
    plot.draw_precision_recall(plotting_results, jet_names)
    plot.draw_loc_delta(plotting_deltas, jet_names)
