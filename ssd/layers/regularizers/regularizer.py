import torch
import torch.nn as nn


class FLOPRegularizer():

    def __init__(self, in_shape, reg_strength=1e-11, threshold=1e-2):
        self.activation_size = torch.Tensor(in_shape[1:])
        self.channels = in_shape[0]
        self.reg_strength = reg_strength
        self.threshold = threshold

    def feature_map_after_pooling(self, A):
        """
        Calculates feature map size after pooling operation.
        Formula: ((Activations−Kernel_Size+2*Padding)/Stride)+1
        """
        K, P, S = 2, 1, 2
        return torch.floor_divide((A - K + 2*P), S) + 1

    def flops_per_block(self, in_channels, in_shape, gammas, depthwise=True):
        """
        Calculate FLOPs for the convolutions with activations.
        Assumes folded BatchNorm.
        """
        K, P, S = 3, 1, 1

        alive = torch.sum(torch.abs(gammas) > self.threshold)
        num_instance_per_filter = ((in_shape[0] - K + 2 * P) / S) + 1
        num_instance_per_filter *= ((in_shape[1] - K + 2 * P) / S) + 1

        if depthwise:
            # [in_C * W * H  * (out_C + K * K)]
            flops = in_channels * num_instance_per_filter * (alive + K * K)
        else:
            # [in_C * W * H  * (out_C * K * K)]
            flops = in_channels * num_instance_per_filter * (alive * K * K)
        # Add activations
        flops += in_channels * num_instance_per_filter
        flops = self.reg_strength*flops*torch.sum(torch.abs(gammas))
        return flops, alive

    def get_regularization(self, net):
        total_flops, c, in_size = 0, self.channels, self.activation_size

        for x, child in enumerate(net.children()):
            # Take only inference batchnorm parameters
            if x > 11:
                break

            cflops = 0
            if not self.is_pooling_layer(child):
                if len(child) == 4:
                    conv_dw = False
                    _, gamma = list(child.named_parameters())[1]

                if len(child) == 6:
                    conv_dw = True
                    _, gamma = list(child.named_parameters())[-3]

                cflops, c = self.flops_per_block(c, in_size, gamma, conv_dw)
            else:
                in_size = self.feature_map_after_pooling(in_size)
            total_flops += cflops

        return total_flops

    def is_pooling_layer(self, layer):
        """
        Checks if layer is a pooling layer.
        """
        return isinstance(layer, nn.AvgPool2d)
