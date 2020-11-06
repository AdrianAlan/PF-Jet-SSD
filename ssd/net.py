import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import itemgetter
from ssd.layers import *
from torch.autograd import Variable


class SSD(nn.Module):

    def __init__(self, phase, base, head, ssd_settings):
        super(SSD, self).__init__()

        self.phase = phase
        self.vgg = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.priorbox = PriorBox()
        self.conf = nn.ModuleList(head[1])
        self.regr = nn.ModuleList(head[2])
        self.num_classes = ssd_settings['n_classes']
        self.top_k = ssd_settings['top_k']
        self.min_confidence = ssd_settings['confidence_threshold']
        self.nms = ssd_settings['nms']
        config = {'min_dim': ssd_settings['input_dimensions'][1:],
                  'feature_maps': ssd_settings['feature_maps'],
                  'steps': ssd_settings['steps'],
                  'size': ssd_settings['object_size']}
        self.priors = Variable(self.priorbox.apply(config))
        self.L2Norm1 = L2Norm(256, 20)
        self.L2Norm2 = L2Norm(256, 20)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input images x"""

        sources, loc, conf, regr = list(), list(), list(), list()

        # Add base network
        for k in range(33):
            x = self.vgg[k](x)
        sources.append(self.L2Norm1(x))
        for k in range(33, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(self.L2Norm2(x))

        # Apply multibox head to source layers
        for (x, l, c, r) in zip(sources, self.loc, self.conf, self.regr):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            regr.append(r(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        regr = torch.cat([o.view(o.size(0), -1) for o in regr], 1)

        # Apply correct output layer
        if self.phase == "test":
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 2),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                regr.view(regr.size(0), -1, 1),
                self.priors.type(type(x.data)),
                self.num_classes,
                self.top_k,
                self.min_confidence,
                self.nms)
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                conf.view(conf.size(0), -1, self.num_classes),
                regr.view(regr.size(0), -1, 1),
                self.priors)
        return output

    def load_weights(self, file_path):
        other, ext = os.path.splitext(file_path)
        if ext == '.pkl' or '.pth':
            self.load_state_dict(torch.load(file_path,
                                 map_location=lambda storage, loc: storage))
            return True
        return False


def vgg(in_channels):
    layers = []
    for v in [32, 32, 'P', 64, 64, 'P', 128, 128, 128, 'P', 256, 256, 256, 'P',
              256, 256, 256]:
        if v == 'P':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                       nn.BatchNorm2d(v),
                       nn.PReLU(v)]
            in_channels = v
    layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
               nn.Conv2d(in_channels, 256, kernel_size=3),
               nn.BatchNorm2d(256),
               nn.PReLU(256),
               nn.Conv2d(256, 256, kernel_size=1),
               nn.BatchNorm2d(256),
               nn.PReLU(256)]
    return layers


def multibox(base, num_classes):
    loc, conf, regr = [], [], []

    base_sources = [27, 47]

    for k, v in enumerate(base_sources):
        loc += [nn.Conv2d(base[v].out_channels, 2,
                          kernel_size=3, padding=1)]
        conf += [nn.Conv2d(base[v].out_channels, num_classes,
                           kernel_size=3, padding=1)]
        regr += [nn.Conv2d(base[v].out_channels, 1,
                           kernel_size=3, padding=1)]

    return (loc, conf, regr)


def build_ssd(phase, ssd_settings, qtype='full'):

    input_dimensions = ssd_settings['input_dimensions']

    base = vgg(input_dimensions[0])
    head = multibox(base, ssd_settings['n_classes'])

    return SSD(phase, base, head, ssd_settings)
