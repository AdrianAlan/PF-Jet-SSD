import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Function
from torch.autograd import Variable


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale, device):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        x *= self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x
