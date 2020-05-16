from __future__ import print_function

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import sys

from sklearn.metrics import average_precision_score, precision_recall_curve
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from time import time
from tqdm import tqdm
from utils import Plotting


def get_data_loader(source_path, batch_size, num_workers, shuffle=False):
    h5 = h5py.File(source_path, 'r')
    generator = CalorimeterJetDataset(hdf5_dataset=h5)
    return torch.utils.data.DataLoader(generator,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers), h5


def test_net(model, dataset, top_k=200, im_size=(300, 300),
             conf_threshold=0.05, overlap_threshold=0.1, jet_classes=[]):

    results = len(jet_classes)*[np.empty((0, 2))]
    inf_time = []

    with torch.no_grad():

        progress_bar = tqdm(total=len(dataset), desc='Evaluating events')
        for data, targets in dataset:

            data = data.cuda()
            t_start = time()
            detections = model(data).data
            t_end = time()
            inf_time.append(t_end-t_start)

            targets = targets[0].cpu().numpy()
            targets[:, 0] *= im_size[0]
            targets[:, 2] *= im_size[0]
            targets[:, 1] *= im_size[1]
            targets[:, 3] *= im_size[1]

            all_detections = np.empty((0, 7))
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]

                # Filter detections above given threshold
                dets = dets[dets[:, 0] > conf_threshold]

                if dets.size(0) == 0:
                    continue

                boxes = dets[:, 1:]
                boxes[:, 0] *= im_size[0]
                boxes[:, 2] *= im_size[0]
                boxes[:, 1] *= im_size[1]
                boxes[:, 3] *= im_size[1]

                scores = dets[:, 0].cpu().numpy()
                labels = np.array([j-1]*len(scores))

                # Detection format: [xmin, ymin, xmax, ymax, label, score, gt]
                class_det = np.hstack((boxes.cpu().numpy(),
                                       labels[:, np.newaxis],
                                       scores[:, np.newaxis],
                                       np.zeros(len(boxes))[:, np.newaxis])
                                      ).astype(np.float32, copy=False)

                all_detections = np.vstack((all_detections, class_det))

            # Sort by confidence
            all_detections = all_detections[(-all_detections[:, -2]).argsort()]

            # Select top k predictions
            all_detections = all_detections[:top_k]

            for t in targets:
                detected = False

                for x, d in enumerate(all_detections):
                    ixmin = np.maximum(t[0], d[0])
                    iymin = np.maximum(t[1], d[1])
                    ixmax = np.minimum(t[2], d[2])
                    iymax = np.minimum(t[3], d[3])

                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    intersection = iw * ih

                    union = ((d[2] - d[0]) * (d[3] - d[1]) +
                             (t[2] - t[0]) * (t[3] - t[1]) - intersection)

                    overlap = intersection / (union + 10e-12)

                    if overlap > overlap_threshold:
                        if d[4] == t[4]:
                            detected = True
                            all_detections[x][6] = 1
                            break

                if not detected:
                    fn = np.hstack((t, [0, 1])).astype(np.float32, copy=False)
                    all_detections = np.vstack((all_detections, fn))

            for c in range(len(jet_classes)):
                class_detections = all_detections[all_detections[:, 4] == c]
                results[c] = np.concatenate((results[c],
                                             class_detections[:, [6, 5]]))

            progress_bar.update(1)

        progress_bar.close()

        it = 1000*np.mean(inf_time)

        ret = []
        for c in range(len(jet_classes)):
            p, r, _ = precision_recall_curve(results[c][:, 0],
                                             results[c][:, 1])
            ap = average_precision_score(results[c][:, 0], results[c][:, 1])
            ret.append((r, p, jet_classes[c], ap))

        return it, ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate Jet Detection Model')
    parser.add_argument('source_path', type=str, help='Model source path')
    parser.add_argument('qtype', type=str,
                        choices={'full', 'ternary', 'binary'},
                        help='Type of quantization')
    parser.add_argument('out_plot_path', type=str, help='Path to output plot')
    parser.add_argument('test_dataset', type=str,
                        help='Path to test dataset')
    parser.add_argument('-c', '--classes', type=int, default=1,
                        help='Number of target classes', dest='num_classes')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of workers', dest='num_workers')
    parser.add_argument('-ot', '--overlap-threshold', type=float,
                        default='0.5', help='IoU threshold',
                        dest='overlap_threshold')
    parser.add_argument('-ct', '--confidence-threshold', type=float,
                        default='0.01', help='Confidence threshold',
                        dest='confidence_threshold')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model_source_path = args.source_path
    plot_name = args.out_plot_path
    num_classes = args.num_classes
    jet_classes = ['H+W', 't']
    im_size = (360, 340)
    top_k = 10

    num_classes = num_classes + 1  # +1 for background
    net = build_ssd('test', num_classes, args.qtype)
    net.load_weights(model_source_path)
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True

    loader, h5 = get_data_loader(args.test_dataset, 1, args.num_workers,
                                 shuffle=False)

    it, results = test_net(net, loader, top_k=top_k, im_size=im_size,
                           conf_threshold=args.confidence_threshold,
                           overlap_threshold=args.overlap_threshold,
                           jet_classes=jet_classes)
    print('Average inference time: {0:.3f} ms'.format(it))
    for _, _, c, ap in results:
        print('Average precision for class {0}: {1:.3f}'.format(c, ap))
    plot = Plotting(save_path=plot_name)
    plot.draw_precision_recall(results)
