import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
import torch
import torch.nn as nn

from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from ssd.layers import *


class Plotting():

    def __init__(self, save_dir='./plots', ref_recall=0.3):

        self.save_dir = save_dir
        self.line_styles = [(0, ()), (0, (2, 2))]
        self.legend = ['Full Precision Network', 'Ternary Weight Network']
        self.loc_legend = ['Q2', r'$\mu$']
        self.ref_recall = ref_recall

        plt.style.use('./plots/ssdjet.mplstyle')
        matplotlib.rcParams["figure.figsize"] = (8.0, 6.0)

        with open('./plots/palette.json') as json_file:
            self.color_palette = json.load(json_file)
        self.colors = [self.color_palette['indigo'],
                       self.color_palette['red'],
                       self.color_palette['green'],
                       self.color_palette['yellow']]
        self.markers = ["v", "D"]

    def get_logo(self):
        return OffsetImage(plt.imread('./plots/hls4mllogo.jpg', format='jpg'),
                           zoom=0.08)

    def draw_loss(self, data_train, data_val, type='full',
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

        fig.savefig('%s/loss-%s' % (self.save_dir, type))
        plt.close(fig)

    def draw_precision_recall(self, data, names):
        """Plots the precision recall curve"""

        fig, ax = plt.subplots()
        plt.xlabel("Recall (TPR)", horizontalalignment='right', x=1.0)
        plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)
        ref_precisions = []

        for q, data_model in enumerate(data):
            shade = 'shade_800' if q else 'shade_500'
            for x, (recall, precision, c, ap) in enumerate(data_model):
                # Helper line
                ref_precision = np.round(
                    precision[(np.abs(recall - self.ref_recall)).argmin()], 2)
                ref_precisions.append(ref_precision)
                ax.plot([0, 0.3], [ref_precision, ref_precision],
                        linestyle=self.line_styles[1],
                        linewidth=0.8,
                        alpha=0.5,
                        color=self.color_palette['grey']['shade_500'])

                recall = np.append(1, recall)
                precision = np.append(0, precision)
                plt.plot(recall, precision,
                         linestyle=self.line_styles[q],
                         color=self.colors[x][shade],
                         label='{0}: {1} jets, AP: {2:.3f}'.format(
                                 self.legend[q], names[c], ap))

        # Helper line c.d.
        plt.xticks(list(plt.xticks()[0]) + [self.ref_recall])
        plt.ylim(0.5, 1)
        plt.yticks(list(set([0.5, 0.7, 0.9, 1] + ref_precisions)))
        ax.plot([0.3, 0.3], [0, np.max(ref_precisions)],
                linestyle=self.line_styles[1],
                alpha=0.5,
                color=self.color_palette['grey']['shade_500'])

        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1))

        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=13)

        ab = AnnotationBbox(self.get_logo(), [0, 1], xybox=(0.12, 1.085),
                            frameon=False)
        ax.add_artist(ab)

        fig.savefig('%s/pr-curve' % self.save_dir)
        plt.close(fig)

    def draw_loc_delta(self, data, names, width=[.1, .05, .03], nbins=15):
        """Plots the localization delta in eta, phi and mass"""

        def get_width(p, w):
            return 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)

        def get_line(x, shade, q, c, mean):
            label = '{0}: {1} jets, {2}'.format(self.legend[q], c,
                                                self.loc_legend[mean])
            if mean:
                return Line2D([0], [0], color=self.colors[x][shade],
                              linestyle=self.line_styles[q], label=label,
                              marker=self.markers[q], lw=0, markersize=4)
            else:
                return Line2D([0], [0], color=self.colors[x][shade],
                              label=label, linestyle=self.line_styles[q])

        for x, (c, w) in enumerate(zip(names, width)):

            for idx, l, n in [(2, r'$\sigma(\eta_{SSD}-\eta_{GT})$', 'eta'),
                              (3, r'$\sigma(\phi_{SSD}-\phi_{GT})$', 'phi'),
                              (4, r'$\frac{|m_{SSD}-m_{GT}|}{m_{GT}}$', 'm')]:

                fig, ax = plt.subplots()
                cst_lgd = []
                plt.xlabel('$p_T$ [GeV]', horizontalalignment='right', x=1.0)
                plt.ylabel(l, horizontalalignment='right', y=1.0)

                for q, d in enumerate(data):
                    shade = 'shade_800' if q else 'shade_200'
                    color = self.colors[x][shade]
                    bins, cls = [], d[d[:, 0] == x]

                    if not q:
                        min_pt, max_pt = np.min(cls[:, 1]), np.max(cls[:, 1])
                        binning = np.logspace(np.log10(min_pt),
                                              np.log10(max_pt), nbins)[1:]

                    bmin = 0
                    for bmax in binning:
                        b = cls[(cls[:, 1] > bmin) & (cls[:, 1] <= bmax)]
                        bins.append(np.abs(b[:, idx]))
                        bmin = bmax
                    cst_lgd.append(get_line(x, shade, q, c, 0))
                    cst_lgd.append(get_line(x, shade, q, c, 1))

                    ax.boxplot(bins,
                               positions=binning,
                               widths=get_width(binning, w),
                               medianprops=dict(linestyle=self.line_styles[q],
                                                color=color),
                               meanprops=dict(marker=self.markers[q],
                                              markeredgecolor=color,
                                              markerfacecolor=color,
                                              markersize=4),
                               sym='',
                               showmeans=True,
                               notch=False,
                               showbox=False,
                               showcaps=False,
                               meanline=False,
                               showfliers=False,
                               whis=0)

                ax.set_xscale("log")
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
                ax.legend(handles=cst_lgd, loc='upper left',
                          bbox_to_anchor=(0, -0.1))

                fig.savefig('%s/delta-%s-%s' % (self.save_dir, c, n))
                plt.close(fig)


class GetResources():

    def __init__(self, net, dummy_input):
        self.net = net
        self.dummy_input = dummy_input

    def zero_ops(self, m, x, y):
        m.total_ops += torch.DoubleTensor([int(0)]).cuda()

    def count_conv(self, m, x, y):
        x = x[0]
        # N x H x W (exclude Cout)
        output = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
        # Cout x Cin x Kw x Kh
        kernel_ops = m.weight.nelement()
        if m.bias is not None:
            # Cout x 1
            kernel_ops += + m.bias.nelement()
        # x N x H x W x Cout x (Cin x Kw x Kh + bias)
        m.total_ops += torch.DoubleTensor([int(output * kernel_ops)]).cuda()

    def count_bn(self, m, x, y):
        x = x[0]
        nelements = x.numel()
        if not m.training:
            nelements = 2 * nelements
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

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
            nn.MaxPool2d: self.zero_ops,
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
        self.net.apply(add_hooks)
        with torch.no_grad():
            self.net(self.dummy_input)
        total_ops = dfs_count(self.net)

        return total_ops
