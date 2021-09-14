import os
import torch
import torch.nn as nn

from torch.quantization import QuantStub, DeQuantStub
from dropblock import DropBlock2D
from torch.cuda.amp import autocast
from ssd.layers import *
from torch.autograd import Variable


class SSD(nn.Module):

    def __init__(self,
                 rank,
                 base,
                 head,
                 ssd_settings,
                 net_channels,
                 ceva=False,
                 inference=False,
                 int8=False,
                 onnx=False):
        super(SSD, self).__init__()

        self.ceva = ceva
        self.inference = inference
        self.int8 = int8
        self.onnx = onnx
        self.rank = rank
        self.mobilenet = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.cnf = nn.ModuleList(head[1])
        self.reg = nn.ModuleList(head[2])
        self.n_classes = ssd_settings['n_classes']
        self.top_k = ssd_settings['top_k']
        self.min_confidence = ssd_settings['confidence_threshold']
        self.nms = ssd_settings['nms']

        if self.int8:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.attention1 = AttentionLayer(torch.device(rank))
            self.l2norm_1 = L2Norm(net_channels[-3], 20, torch.device(rank))

        if self.inference:
            self.priors = Variable(PriorBox().apply(
                {'min_dim': ssd_settings['input_dimensions'][1:],
                 'feature_maps': [ssd_settings['feature_maps'][0]],
                 'steps': [ssd_settings['steps'][0]],
                 'size': ssd_settings['object_size']}, rank))
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()
        elif not self.int8:
            self.l2norm_2 = L2Norm(net_channels[-1], 20, torch.device(rank))
            self.attention2 = AttentionLayer(torch.device(rank))

    def forward(self, x):
        if self.int8:
            return self.forward_pass(x)
        else:
            with autocast():
                return self.forward_pass(x)

    def forward_pass(self, x):
        """Applies network layers and ops on input images x"""
        sources, loc, cnf, reg = list(), list(), list(), list()
        if self.int8:
            x = self.quant(x)

        # Add base network
        for i, layer in enumerate(self.mobilenet):
            x = layer(x)
            if i == 11:
                if self.int8 or self.ceva:
                    sources.append(x)
                else:
                    out = self.l2norm_1(x)
                    out = self.attention1(x)
                    sources.append(out)
            if i == 14:
                if self.int8:
                    sources.append(x)
                else:
                    out = self.l2norm_2(x)
                    out = self.attention2(x)
                    sources.append(out)

        # Apply multibox head to source layers
        for (x, l, c, r) in zip(sources, self.loc, self.cnf, self.reg):
            l, c, r = l(x), c(x), r(x)
            if self.int8:
                l, c, r = self.dequant(l), self.dequant(c), self.dequant(r)
            loc.append(l.permute(0, 2, 3, 1).contiguous())
            cnf.append(c.permute(0, 2, 3, 1).contiguous())
            reg.append(r.permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        cnf = torch.cat([o.view(o.size(0), -1) for o in cnf], 1)
        reg = torch.cat([o.view(o.size(0), -1) for o in reg], 1)

        # Apply correct output layer
        if self.ceva:
            output = (l, c, r)
        elif self.inference and not self.onnx:
            priors = self.priors.type(type(x.data))
            priors = priors.to(torch.device(self.rank))
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 2),
                self.softmax(cnf.view(cnf.size(0), -1, self.n_classes)),
                reg.view(reg.size(0), -1, 1),
                priors,
                self.n_classes,
                self.top_k,
                self.min_confidence,
                self.nms)
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                cnf.view(cnf.size(0), -1, self.n_classes),
                reg.view(reg.size(0), -1, 1))
        return output

    def load_weights(self, file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.pkl' or '.pth':
            state_dict = torch.load(file_path, map_location=lambda s, loc: s)
            self.load_state_dict(state_dict, strict=False)
            return True
        return False


def conv_bn(inp, out, int8):
    if int8:
        act = nn.ReLU()
    else:
        act = nn.PReLU(out)

    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out),
        DropBlock2D(block_size=3, drop_prob=0.1),
        act
    )


def conv_dw(inp, out, int8):
    if int8:
        act_1 = nn.ReLU()
        act_2 = nn.ReLU()
    else:
        act_1 = nn.PReLU(inp)
        act_2 = nn.PReLU(out)

    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False,
                  groups=inp),
        nn.BatchNorm2d(inp),
        act_1,
        nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out),
        act_2
    )


def mobile_net_v1(c, net_channels, int8, inference):
    layers = [conv_bn(c, net_channels[0], int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(net_channels[0], net_channels[1], int8),
              conv_dw(net_channels[1], net_channels[2], int8),
              conv_dw(net_channels[2], net_channels[3], int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(net_channels[3], net_channels[4], int8),
              conv_dw(net_channels[4], net_channels[5], int8),
              conv_dw(net_channels[5], net_channels[6], int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(net_channels[6], net_channels[7], int8),
              conv_dw(net_channels[7], net_channels[8], int8),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
              conv_dw(net_channels[8], net_channels[9], int8),
              conv_dw(net_channels[9], net_channels[10], int8)]
    if inference:
        return layers[:-3]
    return layers


def multibox(n_classes, net_channels, inference):
    loc, cnf, reg = [], [], []

    if inference:
        source_channels = [net_channels[-3]]
    else:
        source_channels = [net_channels[-3], net_channels[-1]]

    for c in source_channels:
        loc += [nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False)]
        cnf += [nn.Conv2d(c, n_classes, kernel_size=3, padding=1, bias=False)]
        reg += [nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=False)]

    return (loc, cnf, reg)


def build_ssd(rank,
              ssd_settings,
              net_channels,
              ceva=False,
              inference=False,
              int8=False,
              onnx=False):

    input_dimensions = ssd_settings['input_dimensions']

    base = mobile_net_v1(input_dimensions[0], net_channels, int8, inference)
    head = multibox(ssd_settings['n_classes'], net_channels, inference)

    return SSD(rank,
               base,
               head,
               ssd_settings,
               net_channels,
               ceva,
               inference,
               int8,
               onnx)
