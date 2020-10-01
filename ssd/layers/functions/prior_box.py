from __future__ import division
import torch

from itertools import product as product
from torch.autograd import Function


class PriorBox(Function):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    @staticmethod
    def forward(ctx, config):

        mean = []
        image_size = config['min_dim']
        feature_maps = config['feature_maps']
        steps = config['steps']
        size = config['size']

        s_k_max_x = size/image_size[1]
        s_k_max_y = size/image_size[0]

        for maps, step in zip(feature_maps, steps):

            for i, j in product(range(maps['eta']), range(maps['phi'])):

                f_k_x = image_size[1] / step['eta']
                f_k_y = image_size[0] / step['phi']

                cx = (i + 0.5) / f_k_x
                cy = (j + 0.5) / f_k_y

                mean += [cx, cy, s_k_max_x, s_k_max_y]

        output = torch.Tensor(mean).view(-1, 4)

        return output
