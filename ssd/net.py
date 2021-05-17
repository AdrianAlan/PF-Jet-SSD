import os
import torch
import torch.nn as nn

from dropblock import DropBlock2D
from torch.cuda.amp import autocast
from ssd.layers import *
from torch.autograd import Variable


class SSD(nn.Module):

    def __init__(self, base, head, ssd_settings, inference, rank, onnx):
        super(SSD, self).__init__()

        self.inference = inference
        self.onnx = onnx
        self.rank = rank
        self.mobilenet = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.cnf = nn.ModuleList(head[1])
        self.reg = nn.ModuleList(head[2])
        self.l2norm_1 = L2Norm(512, 20)
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
            self.l2norm_2 = L2Norm(1024, 20)

        self.priors = Variable(self.priorbox.apply(config, rank))

    @autocast()
    def forward(self, x):
        """Applies network layers and ops on input images x"""

        sources, loc, cnf, reg = list(), list(), list(), list()

        # Add base network
        for i, layer in enumerate(self.mobilenet):
            if self.rank != 'cpu':
                layer = layer.cuda(self.rank)
            x = layer(x)
            if i == 11:
                sources.append(self.l2norm_1(x))
            if i == 14:
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
        if self.onnx:
            output = (
                loc.view(loc.size(0), -1, 2),
                cnf.view(cnf.size(0), -1, self.n_classes),
                reg.view(reg.size(0), -1, 1))
        elif self.inference:
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


def conv_bn(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out),
        DropBlock2D(block_size=3, drop_prob=0.1),
        nn.PReLU(out)
    )


def conv_dw(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False,
                  groups=inp),
        nn.BatchNorm2d(inp),
        nn.PReLU(inp),
        nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out),
        nn.PReLU(out)
    )


def mobile_net_v1(c, inference):
    layers = [conv_bn(c, 32),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(32, 64),
              conv_dw(64, 128),
              conv_dw(128, 128),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(128, 256),
              conv_dw(256, 512),
              conv_dw(512, 512),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(512, 512),
              conv_dw(512, 512),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(512, 1024),
              conv_dw(1024, 1024)]
    if inference:
        return layers[:-3]
    return layers


def multibox(n_classes, inference):
    loc, cnf, reg = [], [], []

    if inference:
        source_channels = [512]
    else:
        source_channels = [512, 1024]

    for c in source_channels:
        loc += [nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False)]
        cnf += [nn.Conv2d(c, n_classes, kernel_size=3, padding=1, bias=False)]
        reg += [nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=False)]

    return (loc, cnf, reg)


def build_ssd(rank, ssd_settings, inference=False, onnx=False):

    input_dimensions = ssd_settings['input_dimensions']

    base = mobile_net_v1(input_dimensions[0], inference)
    head = multibox(ssd_settings['n_classes'], inference)

    return SSD(base, head, ssd_settings, inference, rank, onnx)
