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
        self.line_styles = [(0, (2, 2)), (0, ()), (0, (2, 2))]
        self.legend = ['Full Precision',
                       'Ternary Weight',
                       r'Baseline AK8+$m_{SOFTDROP}$+$\tau_{21}/\tau_{32}$']
        self.shades = ['shade_400', 'shade_900', 'shade_200']
        self.ref_recall = ref_recall

        plt.style.use('./plots/ssdjet.mplstyle')
        matplotlib.rcParams["figure.figsize"] = (8.0, 6.0)

        with open('./plots/palette.json') as json_file:
            self.color_palette = json.load(json_file)
        self.colors = [self.color_palette['indigo'],
                       self.color_palette['red'],
                       self.color_palette['green'],
                       self.color_palette['yellow']]
        self.markers = ["o", "v", "s"]

    def get_logo(self):
        return OffsetImage(plt.imread('./plots/hls4mllogo.jpg', format='jpg'),
                           zoom=0.08)

    def draw_loss(self, data_train, data_val, quantized,
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
                     color=color['shade_800'],
                     label=key)
            plt.plot(val,
                     linestyle=self.line_styles[1],
                     color=color['shade_400'])

        ax.legend()
        if quantized:
            name = 'twn'
        else:
            name = 'fpn'
        fig.savefig('%s/loss-%s' % (self.save_dir, name))
        plt.close(fig)

    def draw_precision_recall(self, results_fp, results_tp, results_base,
                              jet_names):
        """Plots the precision recall curve"""

        fig, ax = plt.subplots()
        for i, results in enumerate([results_fp, results_tp, results_base]):
            model_name = self.legend[i]
            for x, jet_name in enumerate(jet_names):
                score = results[x][:, 3].cpu().numpy()
                truth = results[x][:, 4].cpu().numpy()
                precision, recall, _ = precision_recall_curve(truth, score)
                ap = average_precision_score(truth, score)
                label = '{0}: {1} jets, AP: {2:.3f}'.format(model_name
                                                            jet_name, ap)
                plt.plot(recall[1:],
                         precision[1:],
                         linestyle=self.line_styles[i],
                         color=self.colors[x][self.shades[i]],
                         label=label)

        plt.xlabel('Recall (TPR)', horizontalalignment='right', x=1.0)
        plt.ylabel('Precision (PPV)', horizontalalignment='right', y=1.0)
        plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1))
        fig.savefig('%s/precision-recall-curve' % self.save_dir)
        plt.close(fig)

    def draw_precision_details(self, results_fp, results_tp, results_base,
                               jet_names, nbins=11):
        """Plots precision at fixed recall as a function of pT"""

        legend_helper_network, legend_helper_type = [], []
        for i, name in enumerate(self.legend):
            legend_helper_network.append(Line2D([], [],
                                                linewidth=0,
                                                markersize=5,
                                                marker=self.markers[i],
                                                color='black',
                                                label=name))
        for i, jet in enumerate(jet_names):
            legend_helper_type.append(Line2D([], [],
                                             linewidth=2,
                                             color=self.colors[i]['shade_800'],
                                             label='%s jets' % jet))

        for i, l, n in [(0, r'$\eta$', 'eta'),
                        (1, r'$\phi$', 'phi'),
                        (5, r'$pT [GeV]$', 'pt')]:

            fig, ax = plt.subplots()
            plt.xlabel(l, horizontalalignment='right', x=1.0)
            plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)

            # Fix binning across classes
            if i == 5:
                v = np.array([])
                for x, _ in enumerate(jet_names):
                    v = np.append(v, results_base[x][:, i].cpu().numpy())
                max_pt = np.max(v)
                binning = np.logspace(2, np.log10(max_pt), nbins)[1:]
                ax.set_xlim([100, max_pt*1.2])
                ax.set_xscale("log")
            else:
                binning = np.linspace(0, 1, nbins)[1:]
                ax.set_xlim([0, 1])

            for x, jet_name in enumerate(jet_names):
                for index, result in enumerate([results_fp,
                                                results_tp,
                                                results_base]):
                    color = self.colors[x][self.shades[index]]
                    score = result[x][:, 3].cpu().numpy()
                    truth = result[x][:, 4].cpu().numpy()
                    values = result[x][:, i].cpu().numpy()

                    bmin, v = -np.inf, []
                    for bmax in binning:
                        if binning[-1] == bmax:
                            mask = (values > bmin)
                        else:
                            mask = (values > bmin) & (values <= bmax)
                        s, t = score[mask], truth[mask]
                        if len(s):
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
                            marker=self.markers[index],
                            linewidth=0,
                            markersize=5)

            # Add legend
            plt.gca().add_artist(plt.legend(handles=legend_helper_type,
                                            loc='upper left',
                                            bbox_to_anchor=(0, -0.1)))
            plt.gca().add_artist(plt.legend(handles=legend_helper_network,
                                            loc='upper left',
                                            bbox_to_anchor=(0.2, -0.1)))

            fig.savefig('{}/precision-{}'.format(self.save_dir, n))
            plt.close(fig)

    def draw_loc_delta(self, results_fp, results_tp, results_base,
                       jet_names, nbins=12):
        """Plots the localization and regression error"""
        results_fp = results_fp.cpu().numpy()
        results_tp = results_tp.cpu().numpy()
        results_base = results_base.cpu().numpy()

        # Fix binning across classes
        min_pt, max_pt = np.min(results_fp[:, 1]), np.max(results_fp[:, 1])
        binning = np.logspace(np.log10(min_pt), np.log10(max_pt), nbins)[1:]

        # Fix legend helpers
        legend_helper_network = []
        for i, name in enumerate(self.legend):
            legend_helper_network.append(Line2D([], [],
                                                linewidth=0,
                                                markersize=5,
                                                marker=self.markers[i],
                                                color='black',
                                                label=name))

        legend_helper_type = []
        for i, jet in enumerate(jet_names):
            legend_helper_type.append(Line2D([], [],
                                             linewidth=2,
                                             color=self.colors[i]['shade_800'],
                                             label='%s jets' % jet))

        for i, l, n in [(2, r'$\mu|\eta-\eta^{GEN}|$ [rad]', 'eta'),
                        (3, r'$\mu|\phi-\phi^{GEN}|$ [rad]', 'phi'),
                        (4, r'$\mu\frac{|p_T-p_T^{GEN}|}{p_T^{GEN}}$', 'pt')]:

            fig, ax = plt.subplots()
            plt.xlabel('$p_T^{GEN}$ [GeV/s]', horizontalalignment='right', x=1.0)
            plt.ylabel(l, horizontalalignment='right', y=1.0)
            for x, _ in enumerate(jet_names):

                for q, d in enumerate([results_fp, results_tp, results_base]):
                    color = self.colors[x][self.shades[q]]
                    cls = d[d[:, 0] == x+1]
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
                            marker=self.markers[q],
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

            fig.savefig('%s/delta-%s' % (self.save_dir, n))
            plt.close(fig)

    def draw_barchart(self, x, y, label, ylabel,
                      xlabel='Batch size [events]',
                      save_name='inference'):
        """Plots errobars as a function of batch size"""
        fig, ax = plt.subplots()

        width = 0.35
        groups = np.arange(len(x))

        ax.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
        ax.set_ylabel(ylabel, horizontalalignment='right', y=1.0)
        ax.bar(groups - width/2, y[0], label=label[0], width=width,
               color=self.colors[0]['shade_500'])
        ax.bar(groups + width/2, y[1], label=label[1], width=width,
               color=self.colors[1]['shade_500'])
        ax.set_xticks(groups)
        ax.set_xticklabels(x)
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left')

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
            m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
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


def get_data_loader(hdf5_source_path, batch_size, num_workers,
                    input_dimensions, jet_size, rank=0, shuffle=True,
                    return_baseline=False, return_pt=False, raw=False,
                    qbits=None):
    dataset = CalorimeterJetDataset(rank, hdf5_source_path, input_dimensions,
                                    jet_size, return_baseline=return_baseline,
                                    return_pt=return_pt, raw=raw, qbits=qbits)
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
