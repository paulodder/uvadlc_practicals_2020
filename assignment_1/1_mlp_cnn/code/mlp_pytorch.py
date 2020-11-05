"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import itertools as it


class Optimizer(object):
    def __init__(self, mlp, learning_rate):
        self.mlp = mlp
        self.learning_rate = learning_rate

    def step(self):
        with torch.no_grad():
            for hidden_layer in self.mlp.hidden_layers:
                print(dir(hidden_layer))
                hidden_layer.weight -= self.learning_rate * hidden_layer.grad
                hidden_layer.grad_zero()
        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
        for layer in self.mlp.hidden_layers:
            if not hasattr(layer, "grads"):  # layers like softmax
                continue
            for name, vals in layer.params.items():
                layer.params[name] -= self.learning_rate * layer.grads[name]


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_classes,
        activation_fn=nn.ELU,
        batch_norm=True,
        drop_prob=None,
    ):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
        n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the nnetwork.
        """
        super(MLP, self).__init__()
        modules = []
        shapes = [n_inputs] + n_hidden + [n_classes]
        for i in range(1, len(shapes)):
            modules.append(nn.Linear(shapes[i - 1], shapes[i], bias=True))
            if batch_norm:
                modules.append(nn.BatchNorm1d(shapes[i])) if i < len(
                    shapes
                ) - 1 else None
            if drop_prob:
                modules.append(nn.Dropout(drop_prob)) if i < len(
                    shapes
                ) - 1 else None
            modules.append(activation_fn()) if i < len(shapes) - 1 else None
        self.model = nn.Sequential(*modules)

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
        return self.model(x)


# class MLP(nn.Module):
#     """
#     This class implements a Multi-layer Perceptron in PyTorch.
#     It handles the different layers and parameters of the model.
#     Once initialized an MLP object can perform forward.
#     """

#     def __init__(self, n_inputs, n_hidden, n_classes, activation_layer=nn.ELU):
#         """
#         Initializes MLP object.

#         Args:
#           n_inputs: number of inputs.
#           n_hidden: list of ints, specifies the number of units
#                     in each linear layer. If the list is empty, the MLP
#                     will not have any linear layers, and the model
#                     will simply perform a multinomial logistic regression.
#           n_classes: number of classes of the classification problem.
#                      This number is required in order to specify the
#                      output dimensions of the MLP

#         TODO:
#         Implement initialization of the network.
#         """

#         ########################
#         # PUT YOUR CODE HERE  #
#         #######################
#         super(MLP, self).__init__()
#         hidden_layers = list(
#             it.chain.from_iterable(
#                 [
#                     (nn.Linear(inp, outp), activation_layer())
#                     for inp, outp in zip(
#                         [n_inputs] + n_hidden, n_hidden + [n_classes]
#                     )
#                 ]
#             )
#         )
#         # hidden_layers[-1] = nn.Softmax(1)
#         self.layers = nn.Sequential(*hidden_layers[:-1])
#         print(self.layers)
#         ########################
#         # END OF YOUR CODE    #
#         #######################

#     def forward(self, x):
#         """
#         Performs forward pass of the input. Here an input tensor x is transformed through
#         several layer transformations.

#         Args:
#           x: input to the network
#         Returns:
#           out: outputs of the network

#         TODO:
#         Implement forward pass of the network.
#         """

#         ########################
#         # PUT YOUR CODE HERE  #
#         #######################
#         self.x = x[:]
#         # out = x
#         # for layer in self.hidden_layers:
#         #     out = layer.forward(out)
#         out = self.layers.forward(x)
#         ########################
#         # END OF YOUR CODE    #
#         #######################

#         return out
