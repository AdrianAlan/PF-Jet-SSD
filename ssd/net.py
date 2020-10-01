import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import itemgetter
from ssd.layers import *
from torch.autograd import Variable


class SSD(nn.Module):

    def __init__(self, phase, base, extras, head, ssd_settings):
        super(SSD, self).__init__()

        self.phase = phase
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
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
        self.L2Norm = L2Norm(256, 20)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input images x"""

        sources, loc, conf, regr = list(), list(), list(), list()

        # Add base network
        for k in range(33):
            x = self.vgg[k](x)
        sources.append(self.L2Norm(x))
        for k in range(33, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # Add extra layers
        for k, v in enumerate(self.extras):
            x = v(x)
            if k in [4, 11, 18, 24]:
                sources.append(F.relu(x, inplace=True))

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


def vgg(in_channels, conv, acti):
    layers = []
    layers += [nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
               nn.BatchNorm2d(32),
               acti(32)]
    in_channels = 32

    for v in [32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256,
              'M', 256, 256, 256]:
        if v == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            layers += [conv(in_channels, v, kernel_size=3, padding=1),
                       nn.BatchNorm2d(v),
                       acti(v)]
            in_channels = v

    layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
               conv(in_channels, 512, kernel_size=3),
               nn.BatchNorm2d(512),
               acti(512),
               conv(512, 512, kernel_size=1),
               nn.BatchNorm2d(512),
               acti(512)]
    return layers


def extra_layers(conv, acti):
    return [conv(512, 128, kernel_size=1, padding=1),
            nn.BatchNorm2d(128),
            acti(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            conv(128, 256, kernel_size=1, padding=1),
            nn.BatchNorm2d(256),
            acti(256),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            conv(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            acti(64),
            conv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            acti(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
            conv(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            acti(64),
            conv(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            acti(128),
            conv(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            acti(64),
            conv(64, 128, kernel_size=3)]


def multibox(base, extras, num_mbox, num_classes, conv):
    loc, conf, regr = [], [], []

    base_sources = [27, 47]
    extra_sources = [4, 11, 18, 24]

    for k, v in enumerate(base_sources):
        loc += [conv(base[v].out_channels, num_mbox * 2,
                kernel_size=3, padding=1)]
        conf += [conv(base[v].out_channels, num_mbox * num_classes,
                 kernel_size=3, padding=1)]
        regr += [conv(base[v].out_channels, num_mbox * 1,
                 kernel_size=3, padding=1)]

    for k, v in enumerate(extra_sources):
        loc += [conv(extras[v].out_channels, num_mbox * 2,
                kernel_size=3, padding=1)]
        conf += [conv(extras[v].out_channels, num_mbox * num_classes,
                 kernel_size=3, padding=1)]
        regr += [conv(extras[v].out_channels, num_mbox * 1,
                 kernel_size=3, padding=1)]

    return (loc, conf, regr)


def build_ssd(phase, ssd_settings, qtype='full', num_mbox=1):

    if qtype == 'binary':
        conv = BinaryConv2d
        acti = nn.ReLU(inplace=True)
    elif qtype == 'ternary':
        conv = TernaryConv2d
        acti = nn.PReLU
    else:
        conv = nn.Conv2d
        acti = nn.PReLU

    input_dimensions = ssd_settings['input_dimensions']

    base = vgg(input_dimensions[0], conv, acti)
    extras = extra_layers(conv, acti)
    head = multibox(base, extras, num_mbox, ssd_settings['n_classes'], nn.Conv2d)

    return SSD(phase, base, extras, head, ssd_settings)
