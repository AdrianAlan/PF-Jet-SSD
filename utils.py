import argparse
import h5py
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import simplejson as json
import torch
import torch.nn as nn

from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage
from matplotlib.ticker import FixedLocator as Locator
import matplotlib.font_manager
from sklearn.metrics import average_precision_score, precision_recall_curve
from ssd.generator import CalorimeterJetDataset
from ssd.layers import *
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{avg' + self.fmt + '} ({name})'
        return fmtstr.format(**self.__dict__)


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                    '{0} is not a readable directory'.format(prospective_dir))


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid file'.format(prospective_file))
        else:
            setattr(namespace, self.dest, prospective_file)


class Plotting():

    def __init__(self, save_dir='./plots', ref_recall=0.3):

        self.save_dir = save_dir
        self.legend = ['Baseline', 'FPN', 'TWN', 'INT8']
        self.ref_recall = ref_recall

        plt.style.use('./plots/ssdjet.mplstyle')
        with open('./plots/palette.json') as json_file:
            self.color_palette = json.load(json_file)
        shade = 'shade_700'
        self.colors = [self.color_palette['black'],
                       self.color_palette['red'][shade],
                       self.color_palette['blue'][shade],
                       self.color_palette['yellow'][shade],
                       self.color_palette['green'][shade]]
        self.line_styles = [(0, ()),
                            (0, (1, 4)),
                            (0, (6, 4)),
                            (0, (6, 7, 1, 7))]
        self.markers = ["s", "v", "o"]

    def draw_loss(self,
                  data_train,
                  data_val,
                  name='',
                  keys=['Localization', 'Classification', 'Regression']):
        """Plots the training and validation loss"""

        fig, ax = plt.subplots()
        plt.xlabel("Epoch", horizontalalignment='right', x=1.0)
        plt.ylabel("Loss", horizontalalignment='right', y=1.0)
        plt.yscale("log")

        for x, (train, val, key) in enumerate(zip(data_train, data_val, keys)):
            color = self.colors[x]
            plt.plot(train,
                     linestyle=self.line_styles[0],
                     color=color,
                     label=key)
            plt.plot(val,
                     linestyle=self.line_styles[1],
                     color=color)

        ax.legend()
        plt.savefig('{}/loss-{}'.format(self.save_dir, name))
        plt.close(fig)

    def draw_precision_recall(self,
                              results_base,
                              results_fpn,
                              results_twn,
                              results_int8,
                              jet_names):
        """Plots the precision recall curve"""

        def find_nearest(array, value):
            if array[1] < (value - .01):
                return None
            idx = (np.abs(array - value)).argmin()
            return idx

        fig, ax = plt.subplots()
        results_ap, results_pr3, results_pr5 = [], [], []
        for i, results in enumerate([results_base,
                                     results_fpn,
                                     results_twn,
                                     results_int8]):
            name = self.legend[i]
            scores, truths = [], []
            tmp_ap, tmp_pr3, tmp_pr5 = [], [], []
            for j, jet in enumerate(jet_names):
                truth = results[j][:, 4].numpy()
                score = results[j][:, 3].numpy()
                truths = np.concatenate((truths, truth), axis=None)
                scores = np.concatenate((scores, score), axis=None)
                precision, recall, _ = precision_recall_curve(truth, score)
                ap = average_precision_score(truth, score)
                tmp_ap.append(ap)
                x = find_nearest(recall, 0.3)
                if x is None:
                    tmp_pr3.append(np.nan)
                else:
                    tmp_pr3.append(precision[x])
                x = find_nearest(recall, 0.5)
                if x is None:
                    tmp_pr5.append(np.nan)
                else:
                    tmp_pr5.append(precision[x])
            results_ap.append(tmp_ap)
            results_pr3.append(tmp_pr3)
            results_pr5.append(tmp_pr5)

            precision, recall, _ = precision_recall_curve(truths, scores)
            ap = average_precision_score(truths, scores)
            label = r'{0}'.format(name)
            plt.plot(recall[1:],
                     precision[1:],
                     linestyle=self.line_styles[i],
                     linewidth=.5,
                     markersize=0,
                     color=self.colors[0],
                     label=label)

        plt.xlabel(r'Recall (TPR)', horizontalalignment='right', x=1.0)
        plt.ylabel(r'Precision (PPV)', horizontalalignment='right', y=1.0)
        plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.legend(loc='lower right')
        fig.savefig('{}/Precision-Recall-Curve'.format(self.save_dir))
        plt.close(fig)
        return results_ap, results_pr3, results_pr5

    def draw_precision_details(self, gt, fpn, twn, int8, jet_names, nbins=11):
        """Plots the precision histogram at fixed recall"""
        ylabel = 'PPV@R={}'.format(self.ref_recall)
        xlabels = [r'$\eta$', r'$\phi$ [°]', r'$p_T^{SSD}$ [GeV/s]']
        scales = [0.4, 0.5, 20]
        idxs = [0, 1, 5]
        ax_m = [6, 360, 1]
        ax_s = [3, 0, 0]

        for x, jet_name in enumerate(jet_names):
            fig, axs = plt.subplots(3, 3, figsize=(10.5, 5.4))
            for row, result in enumerate([fpn, twn, int8]):
                for column, (i, l, ax_mul, ax_sub) in enumerate(
                        zip(idxs, xlabels, ax_m, ax_s)):

                    ax = axs[row][column]

                    if row == 2:
                        ax.set_xlabel(l, horizontalalignment='right', x=1.0)
                    if column == 0:
                        ax.set_ylabel(ylabel,
                                      horizontalalignment='right',
                                      y=1.0)

                    # Fix binning across classes
                    if i == 5:
                        pt = gt[gt[:, 0] == x+1][:, 1].numpy()
                        min_pt, max_pt = np.min(pt), np.max(pt)
                        binning = np.logspace(np.log10(min_pt),
                                              np.log10(max_pt),
                                              nbins)[1:]
                        ax.set_xscale("log")
                        ax.set_xlim([min_pt, 1.1*max_pt])
                        ax.set_ylim([0, 1.1])
                    else:
                        binning = np.linspace(0, 1, nbins)[1:]
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1.1])

                    score = result[x][:, 3].numpy()
                    truth = result[x][:, 4].numpy()
                    values = result[x][:, i].numpy()
                    bmin, v = 0, []
                    for bmax in binning:
                        if binning[-1] == bmax:
                            mask = (values > bmin)
                        else:
                            mask = (values > bmin) & (values <= bmax)
                        s, t = score[mask], truth[mask]
                        if len(s) and np.sum(t):
                            p, r, _ = precision_recall_curve(t, s)
                            tmp = p[(np.abs(r - self.ref_recall)).argmin()]
                            v.append(np.round(tmp, 2))
                        else:
                            v.append(np.nan)
                        bmin = bmax

                    if i == 5:
                        xvalues = binning
                    else:
                        xvalues = binning-binning[0]/2

                    ax.plot(xvalues,
                            v,
                            color=self.colors[0],
                            marker=self.markers[0],
                            linewidth=0)

                    if i == 0:
                        ticks = ax.get_xticks()*ax_mul-ax_sub
                        ticks = np.round_(ticks, decimals=2)
                        ax.xaxis.set_major_locator(Locator(ax.get_xticks()))
                        ax.set_xticklabels(ticks)
                    if i == 1:
                        ticks = ax.get_xticks()*ax_mul-ax_sub
                        ticks = ticks.astype(np.int32)
                        ax.xaxis.set_major_locator(Locator(ax.get_xticks()))
                        ax.set_xticklabels(ticks)
                    plt.setp(ax.get_yticklabels(), visible=column == 0)
                    plt.setp(ax.get_xticklabels(), visible=row == 2)

            plt.savefig('{}/Precision-{}'.format(self.save_dir, jet_name))
            plt.close(fig)

    def draw_loc_delta(self, base, fpn, twn, int8, jet_names, nbins=11):
        """Plots the localization and regression error"""
        xlabel = r'$p_T^{GEN}$ [GeV/s]'
        ylabels = [r'$\eta-\eta^{GEN}$',
                   r'$\phi-\phi^{GEN}$ [rad]',
                   r'$|\frac{p_T}{p_T^{GEN}}|$']
        scales = [0.4, 0.5, 20]
        idxs = [2, 3, 4]

        for x, jet_name in enumerate(jet_names):
            fig, axs = plt.subplots(3, 4, figsize=(14, 5.4))
            for row, (idx, ylabel, s) in enumerate(zip(idxs, ylabels, scales)):
                # Fix binning across classes
                pt = base[base[:, 0] == x+1][:, 1].numpy()
                min_pt, max_pt = np.min(pt), np.max(pt)
                binning = np.logspace(np.log10(min_pt),
                                      np.log10(max_pt),
                                      nbins)[1:]
                for column, results in enumerate([base, fpn, twn, int8]):

                    ax = axs[row][column]

                    if row == 2:
                        ax.set_xlabel(xlabel,
                                      horizontalalignment='right',
                                      x=1.0)
                    if column == 0:
                        ax.set_ylabel(ylabel,
                                      horizontalalignment='right',
                                      y=1.0)

                    cls = results[results[:, 0] == x+1].numpy()
                    bmin, v, e = 0, [], []
                    for bmax in binning:
                        b = cls[(cls[:, 1] > bmin) & (cls[:, 1] <= bmax)]
                        if row == 2:
                            absb = np.abs(b[:, idx])
                        else:
                            absb = b[:, idx]
                        if len(absb):
                            v.append(np.mean(absb))
                            e.append(np.std(absb))
                        else:
                            v.append(np.nan)
                            e.append(np.nan)
                        bmin = bmax

                    ax.errorbar(binning,
                                v,
                                yerr=e,
                                ecolor=self.colors[0],
                                color=self.colors[0],
                                marker=self.markers[0],
                                capsize=2,
                                elinewidth=0.5,
                                linewidth=0)

                    if row == 2:
                        ax.set_ylim([0, s])
                        ax.set_yscale("symlog")
                    else:
                        ax.set_ylim([-s, s])

                    ax.set_xlim([min_pt, max_pt*1.2])
                    ax.set_xscale("log")
                    plt.setp(ax.get_yticklabels(), visible=column == 0)
                    plt.setp(ax.get_xticklabels(), visible=row == 2)
            plt.savefig('%s/Delta-%s' % (self.save_dir, jet_name))
            plt.close(fig)

    def draw_barchart(self, x, y, label, ylabel,
                      xlabel='Batch size [events]',
                      save_name='inference'):
        """Plots errobars as a function of batch size"""
        fig, ax = plt.subplots()

        width = 0.11
        groups = np.arange(len(x))

        ax.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
        ax.set_ylabel(ylabel, horizontalalignment='right', y=1.0)
        ax.bar(groups - 0.36, y[0], label=label[0], width=width)
        ax.bar(groups - 0.24, y[1], label=label[1], width=width)
        ax.bar(groups - 0.12, y[2], label=label[2], width=width)
        ax.bar(groups, y[3], label=label[3], width=width)
        ax.bar(groups + 0.12, y[4], label=label[4], width=width)
        ax.bar(groups + 0.24, y[5], label=label[5], width=width)
        ax.bar(groups + 0.36, y[6], label=label[6], width=width)
        ax.set_xticks(groups)
        ax.set_xticklabels(x)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2, fontsize=7)
        fig.savefig('{}/{}'.format(self.save_dir, save_name))
        plt.close(fig)


class GetResources():

    def __init__(self, net, dummy_input):
        self.net = net
        self.dummy_input = dummy_input

    def zero_ops(self, m, x, y):
        m.total_ops += torch.DoubleTensor([int(0)]).cuda()

    def count_bn(self, m, x, y):
        x = x[0]
        nelements = 2 * x.numel()
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

    def count_conv(self, m, x, y):
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
        total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)
        m.total_ops += torch.DoubleTensor([int(total_ops)]).cuda()

    def count_prelu(self, m, x, y):
        x = x[0]
        nelements = x.numel()
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

    def profile(self):
        handler_collection = {}
        types_collection = set()

        register_hooks = {
            nn.Conv2d: self.count_conv,
            nn.BatchNorm2d: self.count_bn,
            nn.PReLU: self.count_prelu,
            nn.AvgPool2d: self.zero_ops
        }

        def add_hooks(m: nn.Module):
            m.register_buffer('total_ops', torch.zeros(1))
            m_type = type(m)

            fn = None
            if m_type in register_hooks:
                fn = register_hooks[m_type]
            if fn is not None:
                handler_collection[m] = (m.register_forward_hook(fn))

            types_collection.add(m_type)

        def dfs_count(module: nn.Module, prefix="\t"):
            total_ops = 0
            for m in module.children():
                if m in handler_collection and not isinstance(
                          m, (nn.Sequential, nn.ModuleList)):
                    ops = m.total_ops.item()
                else:
                    ops = dfs_count(m, prefix=prefix + "\t")
                total_ops += ops
            return total_ops
        self.net.eval()
        self.net.apply(add_hooks)
        with torch.no_grad():
            self.net(self.dummy_input)
        total_ops = dfs_count(self.net)

        return total_ops


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = list(transposed_data[1])
    if len(transposed_data) < 3:
        return inp, tgt
    bsl = list(transposed_data[2])
    slr = list(transposed_data[3])
    return inp, tgt, bsl, slr


def get_data_loader(hdf5_source_path,
                    batch_size,
                    num_workers,
                    input_dimensions,
                    jet_size,
                    rank=0,
                    cpu=False,
                    flip_prob=None,
                    raw=False,
                    return_baseline=False,
                    return_pt=False,
                    shuffle=True):
    dataset = CalorimeterJetDataset(torch.device(rank),
                                    hdf5_source_path,
                                    input_dimensions,
                                    jet_size,
                                    cpu=cpu,
                                    flip_prob=flip_prob,
                                    raw=raw,
                                    return_baseline=return_baseline,
                                    return_pt=return_pt)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      num_workers=num_workers,
                      shuffle=shuffle)


def set_logging(name, filename, verbose):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(filename)
    ch = logging.StreamHandler()

    logger.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    if verbose:
        ch.setLevel(logging.INFO)

    f = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                          datefmt='%m/%d/%Y %I:%M')
    fh.setFormatter(f)
    ch.setFormatter(f)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
