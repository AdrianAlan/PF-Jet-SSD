import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import itemgetter
from ssd.layers import *
from torch.autograd import Variable


class SSD(nn.Module):

    def __init__(self, base, head, ssd_settings, inference):
        super(SSD, self).__init__()

        self.inference = inference
        self.vgg = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.cnf = nn.ModuleList(head[1])
        self.reg = nn.ModuleList(head[2])
        self.l2norm_1 = L2Norm(256, 20)
        self.priorbox = PriorBox()
        self.n_classes = ssd_settings['n_classes']
        self.top_k = ssd_settings['top_k']
        self.min_confidence = ssd_settings['confidence_threshold']
        self.nms = ssd_settings['nms']
        config = {'min_dim': ssd_settings['input_dimensions'][1:],
                  'feature_maps': ssd_settings['feature_maps'],
                  'steps': ssd_settings['steps'],
                  'size': ssd_settings['object_size']}

        if self.inference:
            config['feature_maps'] = [config['feature_maps'][0]]
            config['steps'] = [config['steps'][0]]
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()
        else:
            self.l2norm_2 = L2Norm(256, 20)

        self.priors = Variable(self.priorbox.apply(config))

    def forward(self, x):
        """Applies network layers and ops on input images x"""

        sources, loc, cnf, reg = list(), list(), list(), list()

        # Add base network
        for k in range(33):
            x = self.vgg[k](x)
        sources.append(self.l2norm_1(x))
        if not self.inference:
            for k in range(33, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(self.l2norm_2(x))

        # Apply multibox head to source layers
        for (x, l, c, r) in zip(sources, self.loc, self.cnf, self.reg):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            cnf.append(c(x).permute(0, 2, 3, 1).contiguous())
            reg.append(r(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        cnf = torch.cat([o.view(o.size(0), -1) for o in cnf], 1)
        reg = torch.cat([o.view(o.size(0), -1) for o in reg], 1)

        # Apply correct output layer
        if self.inference:
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 2),
                self.softmax(cnf.view(cnf.size(0), -1, self.n_classes)),
                reg.view(reg.size(0), -1, 1),
                self.priors.type(type(x.data)),
                self.n_classes,
                self.top_k,
                self.min_confidence,
                self.nms)
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                cnf.view(cnf.size(0), -1, self.n_classes),
                reg.view(reg.size(0), -1, 1),
                self.priors)
        return output

    def load_weights(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.pkl' or '.pth':
            state_dict = torch.load(file_path, map_location=lambda s, loc: s)
            self.load_state_dict(state_dict, strict=False)
            return True
        return False


def vgg(c, inference):
    layers = []

    if inference:
        cfg = [32, 32, 'P', 64, 64, 'P', 128, 128, 128, 'P', 256, 256, 256]
    else:
        cfg = [32, 32, 'P', 64, 64, 'P', 128, 128, 128, 'P', 256, 256, 256,
               'P', 256, 256, 256]

    for v in cfg:
        if v == 'P':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            layers += [nn.Conv2d(c, v, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(v),
                       nn.PReLU(v)]
            c = v

    if inference:
        return layers

    layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
               nn.Conv2d(c, 256, kernel_size=3, bias=False),
               nn.BatchNorm2d(256),
               nn.PReLU(256),
               nn.Conv2d(256, 256, kernel_size=1, bias=False),
               nn.BatchNorm2d(256),
               nn.PReLU(256)]i

    return layers


def multibox(n_classes, inference):
    loc, cnf, reg = [], [], []

    if inference:
        source_channels = [256]
    else:
        source_channels = [256, 256]

    for c in source_channels:
        loc += [nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False)]
        cnf += [nn.Conv2d(c, n_classes, kernel_size=3, padding=1, bias=False)]
        reg += [nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=False)]

    return (loc, cnf, reg)


def build_ssd(ssd_settings, inference=False):

    input_dimensions = ssd_settings['input_dimensions']

    base = vgg(input_dimensions[0], inference)
    head = multibox(ssd_settings['n_classes'], inference)

    return SSD(base, head, ssd_settings, inference)
