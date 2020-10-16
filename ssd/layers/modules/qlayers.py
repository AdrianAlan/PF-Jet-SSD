import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = Ternary(self.weight.org)

        out = F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


def Ternary(tensor, delta=None, alpha=None):
    n = tensor[0].nelement()

    if delta is None:
        delta = .7 * tensor.norm(1, 3).sum(2).sum(1).div(n)
        delta = torch.repeat_interleave(delta, n).view(tensor.size())

    x = torch.where(torch.abs(tensor) < delta,
                    torch.zeros_like(tensor),
                    tensor.sign())

    if alpha is None:
        count = torch.abs(x).sum()
        abssum = torch.sum(x*tensor)
        alpha = abssum / count

    return x*alpha.type(torch.FloatTensor)
