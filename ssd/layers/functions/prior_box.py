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
        feature_maps_eta = config['feature_maps_eta']
        feature_maps_phi = config['feature_maps_phi']
        steps_eta = config['steps_eta']
        steps_phi = config['steps_phi']
        size = config['size']

        s_k_max_y = size/image_size[0]
        s_k_max_x = size/image_size[1]

        for k, (f_y, f_x) in enumerate(zip(feature_maps_phi,
                                           feature_maps_eta)):
            for i, j in product(range(f_y), range(f_x)):
                f_k_x = image_size[0] / steps_eta[k]
                f_k_y = image_size[1] / steps_phi[k]

                cx = (j + 0.5) / f_k_x
                cy = (i + 0.5) / f_k_y

                mean += [cx, cy, s_k_max_x, s_k_max_y]

        output = torch.Tensor(mean).view(-1, 4)

        return output
