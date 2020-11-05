"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *
import itertools as it


class Optimizer(object):
    def __init__(self, mlp, learning_rate):
        self.mlp = mlp
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.mlp.hidden_layers:
            if not hasattr(layer, "grads"):  # layers like softmax
                continue
            for name, vals in layer.params.items():
                layer.params[name] -= self.learning_rate * layer.grads[name]


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
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
          neg_slope: negative slope parameter for LeakyReLU

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.hidden_layers = list(
            it.chain.from_iterable(
                [
                    (LinearModule(inp, outp), ELUModule())
                    for inp, outp in zip(
                        [n_inputs] + n_hidden, n_hidden + [n_classes]
                    )
                ]
            )
        )
        self.hidden_layers[-1] = SoftMaxModule()
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
        out = x
        for layer in self.hidden_layers:
            out = layer.forward(out)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in reversed(self.hidden_layers):
            dout = layer.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################
