"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import itertools as it


class PreActResNet(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super(PreActResNet, self).__init__()
        print(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.net(x) + x


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()
        channels_in_out = [
            (n_channels, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 256),
            (256, 256),
            (256, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
        ]

        def return_conv(in_dim, out_dim, kernel_size, stride, padding):
            return (
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                # nn.BatchNorm2d(out_dim),
                # nn.ReLU(),
            )

        def return_maxpool(in_dim, out_dim, kernel_size, stride, padding):
            return (
                nn.MaxPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding,
                ),
            )

        def return_preact(in_dim, out_dim, kernel_size, stride, padding):
            return (
                PreActResNet(in_dim, out_dim, kernel_size, stride, padding),
            )

        layer_types = [
            return_conv,  # 0
            return_preact,
            return_conv,  # 1
            return_maxpool,
            return_preact,
            return_preact,
            return_conv,  # 2
            return_maxpool,
            return_preact,
            return_preact,
            return_conv,  # 3
            return_maxpool,
            return_preact,
            return_preact,
            return_maxpool,
            return_preact,
            return_preact,
            return_maxpool,
        ]
        strides = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        paddings = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        kernel_sizes = [
            (3, 3),
            (3, 3),
            (1, 1),
            (3, 3),
            (3, 3),
            (3, 3),
            (1, 1),
            (3, 3),
            (3, 3),
            (3, 3),
            (1, 1),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        ]
        layers = list(
            it.chain.from_iterable(
                [
                    layer_type(in_dim, out_dim, kernel_size, stride, padding)
                    for layer_type, (
                        in_dim,
                        out_dim,
                    ), kernel_size, stride, padding in zip(
                        layer_types,
                        channels_in_out,
                        kernel_sizes,
                        strides,
                        paddings,
                    )
                ]
            )
        )
        print(
            "\n".join(
                map(
                    str, (zip(layer_types, channels_in_out, strides, paddings))
                )
            )
        )

        layers.extend(
            [
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Flatten(1),
                nn.Linear(512, n_classes),
            ]
        )
        # layers.append(nn.Linear(512, n_classes))
        # layers.append()
        self.layers = nn.Sequential(*layers)
        print(self.layers)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # out = x
        # print("start", x.shape)
        # for layer in self.layers:
        #     out = layer.forward(out)
        #     print(out.shape)
        out = self.layers.forward(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
