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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
        self.legend = ['Full Precision Network',
                       'Ternary Weight Network',
                       r'Baseline: m, $\tau$']
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
        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=13)
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
        plt.xlabel("Recall (TPR)", horizontalalignment='right', x=1.0)
        plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)
        ref_precisions = []

        for i, data_model in enumerate([results_fp, results_tp, results_base]):
            for x, (recall, precision, ap) in enumerate(data_model):
                # Helper line
                ref_precision = np.round(
                    precision[(np.abs(recall - self.ref_recall)).argmin()], 2)
                ref_precisions.append(ref_precision)
                ax.plot([0, 0.3], [ref_precision, ref_precision],
                        linestyle=self.line_styles[0],
                        linewidth=0.8,
                        alpha=0.5,
                        color=self.color_palette['grey']['shade_500'])

                plt.plot(recall, precision,
                         linestyle=self.line_styles[i],
                         color=self.colors[x][self.shades[i]],
                         label='{0}: {1} jets, AP: {2:.3f}'.format(
                                 self.legend[i], jet_names[x], ap))

        # Helper line c.d.
        plt.xticks(list(plt.xticks()[0]) + [self.ref_recall])
        plt.yticks(list(set([0.1, 0.3, 0.5, 0.7, 0.9, 1])))
        plt.xlim(0,1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1))

        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=13)

        ab = AnnotationBbox(self.get_logo(), [0, 1], xybox=(0.12, 1.085),
                            frameon=False)
        ax.add_artist(ab)

        fig.savefig('%s/precision-recall-curve' % self.save_dir)
        plt.close(fig)

    def draw_loc_delta(self, results_fp, results_tp, results_base,
                       jet_names, nbins=12):
        """Plots the localization and regression error"""
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

        for i, l, n in [(2, r'$\mu|\eta_{SSD}-\eta_{GT}|$ [rad]', 'eta'),
                        (3, r'$\mu|\phi_{SSD}-\phi_{GT}|$ [rad]', 'phi'),
                        (4, r'$\mu\frac{|Pt_{SSD}-Pt_{GT}|}{Pt_{GT}}$', 'pt')]:

            fig, ax = plt.subplots()
            plt.xlabel('$p_T$ [GeV]', horizontalalignment='right', x=1.0)
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
                            linestyle=self.line_styles[q],
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

            ax.annotate('CMS',
                        xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                        transform=ax.transAxes,
                        horizontalalignment='left',
                        color=self.color_palette['grey']['shade_900'],
                        fontsize=13,
                        weight='bold')

            ab = AnnotationBbox(self.get_logo(),
                                xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                                box_alignment=(-0.5, 0.3),
                                frameon=False)
            ax.add_artist(ab)

            fig.savefig('%s/delta-%s' % (self.save_dir, n))
            plt.close(fig)

    def draw_errorbar(self, x, y, e, ylabel, name):
        """Plots errobars as a function of batch size"""
        fig, ax = plt.subplots()

        xlabel = 'Batch size [events]'
        ax.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
        ax.set_ylabel(ylabel, horizontalalignment='right', y=1.0)
        ax.set_xscale('log')
        ax.errorbar(x, y, yerr=e,
                    fmt='.',
                    elinewidth=1,
                    markersize=5,
                    color=self.colors[0]['shade_500'])

        cms = ax.text(0, 1.02, 'CMS',
                      weight='bold',
                      transform=ax.transAxes,
                      color=self.color_palette['grey']['shade_900'],
                      fontsize=13)

        plt.gcf().canvas.draw()
        bbox = cms.get_window_extent().inverse_transformed(plt.gca().transData)
        ab = AnnotationBbox(self.get_logo(), (x[0], y[0]),
                            xybox=(1.1*bbox.x1, (bbox.y1-bbox.y0)/2 + bbox.y0),
                            xycoords='data',
                            boxcoords='data',
                            box_alignment=(0., 0.5),
                            frameon=False)
        ax.add_artist(ab)
        fig.savefig('%s/inference-%s' % (self.save_dir, name))
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
    return inp, tgt, bsl


def get_data_loader(hdf5_source_path, batch_size, num_workers,
                    input_dimensions, jet_size, rank=0, shuffle=True,
                    return_baseline=False, return_pt=False, qbits=None):
    dataset = CalorimeterJetDataset(rank, hdf5_source_path, input_dimensions,
                                    jet_size, return_baseline=return_baseline,
                                    return_pt=return_pt, qbits=qbits)
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
