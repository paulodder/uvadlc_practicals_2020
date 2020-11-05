"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from mlp_pytorch import MLP, Optimizer
import cifar10_utils

# from mock import mock
import torch
import torch.nn as nn
import dotenv
from pathlib import Path

DOTENV_KEY2VAL = dotenv.dotenv_values()
PROJECT_DIR = Path(DOTENV_KEY2VAL["PROJECT_DIR"])
FIGURE_DIR = PROJECT_DIR / "1_mlp_cnn/figures"
RESULTS_DIR = PROJECT_DIR / "1_mlp_cnn/param_sweep_results/"
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "100"
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
PLOT_DEFAULT = 0
INPUT_SIZE = 1024 * 3
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"
STORE_PROGRESS_DEFAULT = True
FLAGS = None
OPTIMIZER_DEFAULT = "SGD"
ACTIVATION_DEFAULT = "ELU"
UNIT_VARIANCE_DEFAULT = 0
OPTIM_NAME2OBJ = {
    "ADAM": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
ACTIVATION_NAME2OBJ = {"ELU": nn.ELU}


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
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the
    whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [
            int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units
        ]
    else:
        dnn_hidden_units = []

        # neg_slope = FLAGS.neg_slope

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    accs, losses = [], []
    name2dset = cifar10_utils.get_cifar10(
        data_dir=str(
            PROJECT_DIR / "1_mlp_cnn/code" / "cifar10/cifar-10-batches-py/"
        ),
        validation_size=5000,
        unit_variance=FLAGS.unit_variance,
    )
    train_handler = name2dset["train"]
    loss = nn.CrossEntropyLoss()
    mlp = MLP(INPUT_SIZE, dnn_hidden_units, 10)  # NEG_SLOPE_DEFAULT
    nof_steps = 0
    # optimizer = Optimizer(mlp, learning_rate=FLAGS.learning_rate)
    FLAGS.optimizer
    optimizer = OPTIM_NAME2OBJ[FLAGS.optimizer](
        mlp.parameters(), lr=FLAGS.learning_rate
    )
    X_val, y_val = (
        name2dset["validation"].images,
        name2dset["validation"].labels,
    )
    X_val = X_val.reshape(X_val.shape[0], INPUT_SIZE)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val).argmax(1)
    while nof_steps <= FLAGS.max_steps:
        optimizer.zero_grad()
        mlp.train()
        x_train, y_train = train_handler.next_batch(FLAGS.batch_size)
        x_train = x_train.reshape(FLAGS.batch_size, INPUT_SIZE)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train.argmax(1))
        preds = mlp(x_train)
        loss_output = loss(preds, y_train)
        loss_output.backward()
        optimizer.step()
        if nof_steps % FLAGS.eval_freq == 0:
            with torch.no_grad():
                # optimizer.zero_grad()
                y_pred = mlp.forward(X_val)
                acc = accuracy(y_pred, y_val)
                accs.append(acc)
                losses.append(float(loss(y_pred, y_val).detach()))
                print(f"{nof_steps} batches:\tAccuracy {acc}")
        nof_steps += 1
    if FLAGS.store_progress:
        pf = param2fname_prefix(FLAGS)
        with torch.no_grad():
            X_test, y_test = (
                name2dset["test"].images,
                name2dset["test"].labels,
            )
            X_test = X_test.reshape(X_test.shape[0], INPUT_SIZE)
            X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test).argmax(1)
            y_pred = mlp.forward(X_test)
            test_accuracy = accuracy(y_pred, y_test)
            res = {
                "val_accuracy": list(accs),
                "losses": list(losses),
                "test_accuracy": test_accuracy,
            }
        print(res)
        if FLAGS.param_num is not None:
            out_dir = RESULTS_DIR / str(FLAGS.param_num)
            out_dir.mkdir(exist_ok=True)
        else:
            out_dir = RESULTS_DIR
        with open(out_dir / f"{pf}.json", "w") as f:
            f.write(json.dumps(res))

    if FLAGS.plot:
        plot_loss_accs(losses, accs)

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


def param2fname_prefix(flags):
    return "_".join(
        (
            ["pytorch_impl_"]
            + [
                f"{k}={v}"
                for k, v in sorted(
                    vars(flags).items(), key=lambda key_val: key_val[0]
                )
                if k != "data_dir"
            ]
        )
    )


# def parse_args():
#     FLAGS = mock.Mock()
#     # parser.dnn_hidden_units_default
#     FLAGS.batch_size = 10
#     FLAGS.max_steps = MAX_STEPS_DEFAULT
#     FLAGS.learning_rate = LEARNING_RATE_DEFAULT
#     FLAGS.eval_freq = EVAL_FREQ_DEFAULT
#     FLAGS.data_dir = PROJECT_DIR / "1_mlp_cnn/code/cifar10"
#     FLAGS.dnn_hidden_units = "100"
#     return FLAGS, None


def plot_loss_accs(losses, accs):
    plt.clf()
    fig, ax = plt.subplots(2)
    x = [i * int(FLAGS.eval_freq) for i in range(len(losses))]
    for ind, (name, vals) in enumerate(
        zip(["loss", "accuracy"], [losses, accs])
    ):
        ax[ind].plot(x, vals, label=name)
        ax[ind].set_xlabel("# batches trained on")
        ax[ind].set_ylabel(name)
    # fig.suptitle("loss and accuracy plots
    fname = f"{param2fname_prefix(FLAGS)}_loss_accuracy.png"
    fpath = FIGURE_DIR / fname
    print(f"saving to {fpath}")
    plt.savefig(fpath, tight_layout=True, bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma separated list of number of units in each hidden layer",
    )
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
        "--store_progress",
        type=int,
        default=STORE_PROGRESS_DEFAULT,
        help="Store progress",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=OPTIMIZER_DEFAULT,
        help=f"Optimizer to use ({', '.join(OPTIM_NAME2OBJ.keys())})",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=ACTIVATION_DEFAULT,
        help=f"Activation function to use ({', '.join(ACTIVATION_NAME2OBJ.keys())})",
    )
    parser.add_argument(
        "--unit_variance",
        type=int,
        default=UNIT_VARIANCE_DEFAULT,
        help=f"Preprocess data so it has unit covariance",
    )
    parser.add_argument(
        "--param_num",
        type=int,
        default=-1,
        help=f"Number of parameter sweep, affects storage directory",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    # Command line arguments
    FLAGS, unparsed = parse_args()

    main()
