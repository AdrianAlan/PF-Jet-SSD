import os
import torch
import torch.nn as nn

from torch.cuda.amp import autocast
from ssd.layers import *
from ssd.qutils import uniform_quantization
from torch.autograd import Variable

from collections import namedtuple

Bottleneck = namedtuple('Bottleneck', ['stride', 'depth', 'num', 't'])


class _bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4,
                 downsample=None):
        super(_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu3 = nn.PReLU(planes * expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class SSD(nn.Module):

    def __init__(self, base, head, ssd_settings, inference, rank, onnx):
        super(SSD, self).__init__()

        self.inference = inference
        self.onnx = onnx
        self.rank = rank
        self.resnet = nn.ModuleList(base)
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
        for i, layer in enumerate(self.resnet):
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
            for o in [self.resnet]:
                for m in o.modules():
                    if isinstance(m, nn.Conv2d):
                        if m.in_channels == 3:
                            tmp = m.weight.data.clone()
                            m.weight.data.copy_(uniform_quantization(tmp, 16))
            for o in [self.loc, self.cnf, self.reg]:
                for m in o.modules():
                    if isinstance(m, nn.Conv2d):
                        tmp = m.weight.data.clone()
                        m.weight.data.copy_(uniform_quantization(tmp, 16))
            return True
        return False


def resnet(c, inference):
    conv_defs = [Bottleneck(stride=1, depth=32, num=2, t=4),
                 Bottleneck(stride=1, depth=64, num=2, t=4),
                 Bottleneck(stride=1, depth=128, num=2, t=4),
                 Bottleneck(stride=1, depth=256, num=2, t=4)]
    depth_multiplier = 1.0
    min_depth = 8
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = [nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
              nn.BatchNorm2d(32),
              nn.PReLU(32),
              nn.AvgPool2d(kernel_size=2, stride=2, padding=1)]
    in_ch = 32
    for conv_def in conv_defs:
        if conv_def.stride != 1 or in_ch != depth(conv_def.depth * conv_def.t):
            _downsample = nn.Sequential(
                nn.Conv2d(in_ch, depth(conv_def.depth * conv_def.t),
                          kernel_size=1, stride=conv_def.stride, bias=False),
                nn.BatchNorm2d(depth(conv_def.depth * conv_def.t)),
            )
        for n in range(conv_def.num):
            (s, d) = (conv_def.stride, _downsample) if n == 0 else (1, None)
            layers += [_bottleneck(in_ch, depth(conv_def.depth), s,
                                   conv_def.t, d)]
            in_ch = depth(conv_def.depth * conv_def.t)
        layers += [nn.AvgPool2d(kernel_size=2, s=2, padding=1)]

    if inference:
        return layers[:-4]
    return layers[:-1]


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

    base = resnet(input_dimensions[0], inference)
    head = multibox(ssd_settings['n_classes'], inference)

    return SSD(base, head, ssd_settings, inference, rank, onnx)
