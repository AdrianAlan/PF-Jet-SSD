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
        self.colors = [self.color_palette['red'][shade],
                       self.color_palette['blue'][shade],
                       self.color_palette['yellow'][shade],
                       self.color_palette['green'][shade]]
        self.line_styles = ['solid', (0, (1, 3)), (0, (5, 5))]
        self.markers = ["o", "v", "s"]

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

        fig, ax = plt.subplots()
        for i, results in enumerate([results_base,
                                     results_fpn,
                                     results_twn,
                                     results_int8]):
            name = self.legend[i]
            for j, jet in enumerate(jet_names):
                score = results[j][:, 3].numpy()
                truth = results[j][:, 4].numpy()
                precision, recall, _ = precision_recall_curve(truth, score)
                ap = average_precision_score(truth, score)
                label = '{0}: {1} jets, AP: {2:.3f}'.format(name, jet, ap)
                plt.plot(recall[1:],
                         precision[1:],
                         linestyle=self.line_styles[j],
                         color=self.colors[i],
                         label=label)

        plt.xlabel('Recall (TPR)', horizontalalignment='right', x=1.0)
        plt.ylabel('Precision (PPV)', horizontalalignment='right', y=1.0)
        plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.savefig('{}/precision-recall-curve'.format(self.save_dir))
        plt.close(fig)

    def draw_precision_details(self,
                               results_base,
                               results_fpn,
                               results_twn,
                               results_int8,
                               jet_names,
                               nbins=11):
        """Plots the precision histogram at fixed recall"""

        legend_helper_network, legend_helper_type = [], []
        for i, jet in enumerate(jet_names):
            legend_helper_network.append(Line2D([], [],
                                                linewidth=0,
                                                markersize=4,
                                                marker=self.markers[i],
                                                color='black',
                                                label='{} jets'.format(jet)))

        for i in range(3):
            legend_helper_type.append(Line2D([], [],
                                             linewidth=2,
                                             color=self.colors[i+1],
                                             label=self.legend[i+1]))

        for i, l, name, mul, sub in [(0, r'$\eta$', 'eta', 6, 3),
                                     (1, r'$\phi$ [$\degree$]', 'phi', 360, 0),
                                     (5, r'$p_T^{SSD}$ [GeV/s]', 'pt', 1, 0)]:

            fig, ax = plt.subplots()
            plt.xlabel(l, horizontalalignment='right', x=1.0)
            plt.ylabel('Precision (PPV) @ TPR={}'.format(self.ref_recall),
                       horizontalalignment='right',
                       y=1.0)

            # Fix binning across classes
            if i == 5:
                pt = results_base[:, 1].numpy()
                min_pt, max_pt = np.min(pt), np.max(pt)
                binning = np.logspace(np.log10(min_pt),
                                      np.log10(max_pt),
                                      nbins)[1:]
                ax.set_xscale("log")
            else:
                binning = np.linspace(0, 1, nbins)[1:]
                ax.set_xlim([0, 1])

            for x, _ in enumerate(jet_names):
                for index, result in enumerate([results_fpn,
                                                results_twn,
                                                results_int8]):
                    color = self.colors[index+1]
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

                    ax.plot(xvalues, v,
                            color=color,
                            marker=self.markers[x],
                            linewidth=0,
                            markersize=4)
            if i == 0:
                ticks = (ax.get_xticks()*mul-sub)
                ticks = np.round_(ticks, decimals=2)
                plt.xticks(ax.get_xticks(), ticks)
            if i == 1:
                ticks = (ax.get_xticks()*mul-sub)
                ticks = ticks.astype(np.int32)
                plt.xticks(ax.get_xticks(), ticks)

            # Add legend
            plt.gca().add_artist(plt.legend(handles=legend_helper_type,
                                            loc='upper left',
                                            bbox_to_anchor=(0, -0.1)))
            plt.gca().add_artist(plt.legend(handles=legend_helper_network,
                                            loc='upper left',
                                            bbox_to_anchor=(0.2, -0.1)))

            plt.savefig('{}/precision-{}'.format(self.save_dir, name))
            plt.close(fig)

    def draw_loc_delta(self,
                       results_base,
                       results_fpn,
                       results_twn,
                       results_int8,
                       jet_names,
                       nbins=11):
        """Plots the localization and regression error"""

        # Fix binning across classes
        pt = results_base[:, 1].numpy()
        min_pt, max_pt = np.min(pt), np.max(pt)
        binning = np.logspace(np.log10(min_pt), np.log10(max_pt), nbins)[1:]

        # Fix legend helpers
        legend_helper_network, legend_helper_type = [], []
        for i, jet in enumerate(jet_names):
            legend_helper_network.append(Line2D([], [],
                                                linewidth=0,
                                                markersize=4,
                                                marker=self.markers[i],
                                                color='black',
                                                label='{} jets'.format(jet)))

        for i in range(4):
            legend_helper_type.append(Line2D([], [],
                                             linewidth=2,
                                             color=self.colors[i],
                                             label=self.legend[i]))

        for i, l, n in [(2, r'$\mu|\eta-\eta^{GEN}|$', 'eta'),
                        (3, r'$\mu|\phi-\phi^{GEN}|$ [rad]', 'phi'),
                        (4, r'$\mu\frac{|p_T-p_T^{GEN}|}{p_T^{GEN}}$', 'pt')]:

            fig, ax = plt.subplots()
            plt.xlabel('$p_T^{GEN}$ [GeV/s]',
                       horizontalalignment='right',
                       x=1.0)
            plt.ylabel(l, horizontalalignment='right', y=1.0)
            for x, _ in enumerate(jet_names):

                for index, results in enumerate([results_base,
                                                 results_fpn,
                                                 results_twn,
                                                 results_int8]):
                    color = self.colors[index]
                    cls = results[results[:, 0] == x+1].numpy()
                    bmin, v = 0, []
                    for bmax in binning:
                        b = cls[(cls[:, 1] > bmin) & (cls[:, 1] <= bmax)]
                        absb = np.abs(b[:, i])
                        if len(absb):
                            v.append(np.mean(absb))
                        else:
                            v.append(np.nan)
                        bmin = bmax

                    ax.plot(binning, v,
                            color=color,
                            marker=self.markers[x],
                            linewidth=0,
                            markersize=5)
            ax.set_xlim([min_pt, max_pt*1.2])

            # Change to log scale
            ax.set_yscale("log")
            ax.set_xscale("log")

            # Add legends
            plt.gca().add_artist(plt.legend(handles=legend_helper_type,
                                            loc='upper left',
                                            bbox_to_anchor=(0, -0.1)))
            plt.gca().add_artist(plt.legend(handles=legend_helper_network,
                                            loc='upper left',
                                            bbox_to_anchor=(0.2, -0.1)))

            plt.savefig('%s/delta-%s' % (self.save_dir, n))
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
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2)
        fig.tight_layout()
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
