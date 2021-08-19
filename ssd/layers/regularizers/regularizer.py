import torch
import torch.nn as nn


class FLOPRegularizer():

    def __init__(self, in_shape, device, strength=1e-10):
        self.activation_size = torch.Tensor(in_shape[1:])
        self.channels = in_shape[0]
        self.strength = strength
        self.device = device

    def feature_map_after_pooling(self, A):
        """
        Calculates feature map size after pooling operation.
        Formula: ((Activations−Kernel_Size+2*Padding)/Stride)+1
        """
        K, P, S = 2, 1, 2
        return torch.floor_divide((A - K + 2*P), S) + 1

    def flops_per_block(self,
                        in_channels,
                        out_channels,
                        in_shape,
                        depthwise=True):
        """
        Calculate FLOPs for the convolutions with activations.
        Assumes folded BatchNorm.
        """
        K, P, S = 3, 1, 1
        flops = torch.tensor(0.).to(self.device)

        num_instance_per_filter = ((in_shape[0] - K + 2 * P) / S) + 1
        num_instance_per_filter *= ((in_shape[1] - K + 2 * P) / S) + 1

        d = in_channels * num_instance_per_filter
        if depthwise:
            # [in_C * W * H  * (out_C + K * K)]
            flops += d * (out_channels + K * K)
        else:
            # [in_C * W * H  * (out_C * K * K)]
            flops += d * (out_channels * K * K)
        # Add activations
        flops += in_channels * num_instance_per_filter
        return flops

    def get_regularization(self, net):
        reg = torch.tensor(0.).to(self.device)
        in_channels = self.channels
        in_size = self.activation_size

        for x, child in enumerate(net.children()):

            # Take only inference batchnorm parameters
            if x > 11:
                break

            if not self.is_pooling_layer(child):
                if len(child) == 4:
                    conv_dw = False
                    _, gamma = list(child.named_parameters())[1]

                if len(child) == 6:
                    conv_dw = True
                    _, gamma = list(child.named_parameters())[-3]

                out_channels = len(gamma)

                flops = self.flops_per_block(in_channels,
                                             out_channels,
                                             in_size,
                                             conv_dw)
                reg += flops * torch.sum(torch.abs(gamma)) * self.strength
                in_channels = len(gamma)
            else:
                in_size = self.feature_map_after_pooling(in_size)

        return reg

    def is_pooling_layer(self, layer):
        """
        Checks if layer is a pooling layer.
        """
        return isinstance(layer, nn.AvgPool2d)
