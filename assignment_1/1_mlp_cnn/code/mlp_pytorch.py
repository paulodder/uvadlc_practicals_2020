"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from custom_layernorm import CustomLayerNormAutograd
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
        activation_func=nn.ELU,
        batch_normalization=False,
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
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        hidden_layers = list(
            it.chain.from_iterable(
                [
                    (
                        nn.Linear(inp, outp),
                        nn.BatchNorm1d(outp),
                        # CustomLayerNormAutograd(outp),
                        activation_func(),
                    )
                    if batch_normalization
                    else (nn.Linear(inp, outp), activation_func(),)
                    for inp, outp in zip(
                        [n_inputs] + n_hidden, n_hidden + [n_classes]
                    )
                ]
            )
        )
        if batch_normalization:
            hidden_layers = hidden_layers[:-2]
        else:
            hidden_layers = hidden_layers[:-1]
        self.layers = nn.Sequential(*hidden_layers)
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
        self.x = x[:]
        # out = x
        # for layer in self.hidden_layers:
        #     out = layer.forward(out)
        out = self.layers.forward(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
