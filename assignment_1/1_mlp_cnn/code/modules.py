"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = dict()
        self.grads = dict()
        for size, name in [
            ((out_features, in_features), "weight"),
            (out_features, "bias"),
        ]:
            self.params[name] = np.random.normal(0, scale=0.0001, size=size)
            self.grads[name] = np.zeros(shape=size)
        self.params["bias"] = np.zeros(shape=out_features)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args: x: input to the module Returns: out: output of the module

        TODO: Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can
        be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        # print(x.shape, self.params["weight"].shape, self.params["bias"].shape)
        return (x @ self.params["weight"].T) + self.params["bias"].reshape(
            1, -1
        )

        # raise NotImplementedError

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with
        respect to layer parameters in self.grads['weight'] and
        self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout @ self.params["weight"]
        self.grads["weight"] = dout.T @ self.x  # ?
        self.grads["bias"] = dout.T @ np.ones(dout.shape[0])

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        out = np.exp(x - x.max(1).reshape(-1, 1))
        out /= out.sum(1).reshape(-1, 1)
        self.y = out[:]
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = self.y * (dout - (dout * self.y).sum(1).reshape(-1, 1))
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        S = x.shape[0]
        out = 1 / S * (-(np.log(x) * y)).sum()
        ########################
        # END OF YOUR CODE    #
        #######################
        self.out = out
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        S = x.shape[0]
        dx = ((-1 / (S)) * y) / (x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # self.mask = x[:]
        self.exp = np.exp(x)
        self.x = x
        mask = x < 0
        out = np.where(mask, self.exp - 1, x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # x_der = self.x
        # print(x_der)
        # x_der[x_der >= 0] = 1
        # print(x_der)
        # mask = self.x < 0
        # x_der[mask] = self.exp[mask]
        # print(x_der)
        dx = dout * np.where(self.x < 0, self.exp, np.ones_like(self.x))
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
