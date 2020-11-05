"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = "ADAM"
PLOT_DEFAULT = 1
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"
ACTIVATION_DEFAULT = "ELU"
FLAGS = None
OPTIM_NAME2OBJ = {
    "ADAM": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    acc = int((predictions.argmax(1) == targets).sum()) / len(targets)
    ########################
    # END OF YOUR CODE    #
    #######################

    return acc


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    accs, losses = [], []

    convnet = ConvNet(3, 10,)
    optimizer = OPTIM_NAME2OBJ[FLAGS.optimizer](convnet.parameters())
    name2dset = cifar10_utils.get_cifar10()
    train_handler = name2dset["train"]
    loss = nn.CrossEntropyLoss()
    nof_steps = 0
    X_test, y_test = name2dset["test"].images, name2dset["test"].labels
    # X_test = X_test# .reshape(X_test.shape[0], INPUT_SIZE)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).argmax(1)
    while nof_steps < FLAGS.max_steps:
        # print("step")
        optimizer.zero_grad()
        convnet.zero_grad()
        x_train, y_train = train_handler.next_batch(FLAGS.batch_size)
        # x_train = x_train.reshape(FLAGS.batch_size, INPUT_SIZE)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train.argmax(1))
        preds = convnet.forward(x_train)
        loss_output = loss(preds, y_train)
        loss_output.backward()
        optimizer.step()
        if (nof_steps % FLAGS.eval_freq) == 0:
            with torch.no_grad():
                y_pred = convnet.forward(X_test)
                acc = accuracy(y_pred, y_test)
                accs.append(acc)
                losses.append(loss(y_pred, y_test))
                print(f"{nof_steps} batches:\tAccuracy {acc}")
        nof_steps += 1

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + " : " + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
        help="Number of steps to run trainer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Batch size to run trainer.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
        help="Frequency of evaluation on the test set",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR_DEFAULT,
        help="Directory for storing input data",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=PLOT_DEFAULT,
        help="Plot loss/accuracy plot",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=OPTIMIZER_DEFAULT,
        help="Optimizer to use",
    )
    # parser.add_argument(
    #     "--activation",
    #     type=str,
    #     default=ACTIVATION_DEFAULT,
    #     help="Optimizer to use",
    # )

    return parser.parse_known_args()


if __name__ == "__main__":
    # Command line arguments
    FLAGS, unparsed = parse_args()

    main()
