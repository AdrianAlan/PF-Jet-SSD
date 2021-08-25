import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self, device):
        super(AttentionLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        a = self.avg_pool(x)

        # Two different branches of ECA module
        a = self.conv1d(a.squeeze(-1).transpose(-1, -2))
        a = a.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        a = self.sigmoid(a)

        return x * a.expand_as(x)
