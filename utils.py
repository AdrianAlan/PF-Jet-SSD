import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
import torch
import torch.nn as nn

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from ssd.layers import *


class Plotting():

    def __init__(self, save_path, ref_recall=0.3):

        self.save_path = save_path
        self.line_styles = [(0, ()), (0, (5, 2)), (0, (2, 2))]
        self.legend = ['Full Precision Network', 'Ternary Weight Network']
        self.ref_recall = ref_recall

        plt.style.use('plots/ssdjet.mplstyle')
        matplotlib.rcParams["figure.figsize"] = (8.0, 6.0)

        with open('./data/palette.json') as json_file:
            self.color_palette = json.load(json_file)
        self.colors = [self.color_palette['indigo'],
                       self.color_palette['red'],
                       self.color_palette['green']]

    def draw_loss(self, data_train, data_val, keys):
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

        fig.savefig(self.save_path)
        plt.close(fig)

    def draw_precision_recall(self, data):
        """Plots the precision recall curve"""

        fig, ax = plt.subplots()
        plt.xlabel("Recall (TPR)", horizontalalignment='right', x=1.0)
        plt.ylabel("Precision (PPV)", horizontalalignment='right', y=1.0)
        ref_precisions = []

        for q, data_model in enumerate(data):
            shade = 'shade_800' if q else 'shade_500'
            for x, (recall, precision, jet, ap) in enumerate(data_model):
                # Helper line
                ref_precision = np.round(
                    precision[(np.abs(recall - self.ref_recall)).argmin()], 2)
                ref_precisions.append(ref_precision)
                ax.plot([0, 0.3], [ref_precision, ref_precision],
                        linestyle=self.line_styles[2],
                        linewidth=0.8,
                        alpha=0.5,
                        color=self.color_palette['grey']['shade_500'])

                recall = np.append(1, recall)
                precision = np.append(0, precision)
                plt.plot(recall, precision,
                         linestyle=self.line_styles[q],
                         color=self.colors[x][shade],
                         label='{0}: {1} jets, AP: {2:.3f}'.format(
                                 self.legend[q], jet, ap))

        # Helper line c.d.
        plt.xticks(list(plt.xticks()[0]) + [self.ref_recall])
        plt.yticks([0, 1] + ref_precisions)
        ax.plot([0.3, 0.3], [0, np.max(ref_precisions)],
                linestyle=self.line_styles[2],
                alpha=0.5,
                color=self.color_palette['grey']['shade_500'])

        ax.legend(loc='upper center', bbox_to_anchor=(0.25, -0.1))

        ax.text(0, 1.02, 'CMS',
                weight='bold',
                transform=ax.transAxes,
                color=self.color_palette['grey']['shade_900'],
                fontsize=13)

        logo = OffsetImage(plt.imread('./plots/hls4mllogo.jpg', format='jpg'),
                           zoom=0.08)
        ab = AnnotationBbox(logo, [0, 1], xybox=(0.12, 1.085), frameon=False)
        ax.add_artist(ab)

        fig.savefig(self.save_path)
        plt.close(fig)


class GetResources():

    def __init__(self, net, dummy_input):
        self.net = net
        self.dummy_input = dummy_input

    def zero_ops(self, m, x, y):
        m.total_ops += torch.DoubleTensor([int(0)]).cuda()

    def count_convNd(self, m, x, y):
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

    def count_relu(self, m, x, y):
        x = x[0]
        nelements = x.numel()
        m.total_ops += torch.DoubleTensor([int(nelements)]).cuda()

    def profile(self):
        handler_collection = {}
        types_collection = set()

        register_hooks = {
            nn.Conv2d: self.count_convNd,
            TernaryConv2d: self.count_convNd,
            BinaryConv2d: self.count_convNd,
            nn.BatchNorm2d: self.count_bn,
            nn.PReLU: self.count_relu,
            nn.MaxPool2d: self.zero_ops
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

        def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
            total_ops = 0
            for m in module.children():
                if m in handler_collection and not isinstance(
                          m, (nn.Sequential, nn.ModuleList)):
                    m_ops = m.total_ops.item()
                else:
                    m_ops = dfs_count(m, prefix=prefix + "\t")
                total_ops += m_ops
            return total_ops

        self.net.apply(add_hooks)
        with torch.no_grad():
            self.net(*(self.dummy_input, ))
        total_ops = dfs_count(self.net)

        return total_ops
