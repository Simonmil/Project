import numpy as np


class NeuralNetwork:

    def __init__(self, n_input, n_hidden, hidden_types, n_output):
        """
        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_hidden: list of int
            Each item in the list is the size of the hidden layer
            at the corresponding index.
        hidden_types: list of str
            Each item in the list describes the activation function in
            the corresponding hidden layer.
            Supported values are:
                - 'relu':    ReLu activation
                - 'sigmoid': Sigmoid activation
                - 'linear':  Linear activation
            The length of this list should be the same as the length of the n_hidden list.
        n_output : int
            Number of output neurons
        """
        # TODO your code here

    def train(self, x, y, learning_rate, n_epochs):
        """
        Train the neural network by backpropagation,
        using (full-batch) stochastic gradient descent.
        Parameters
        ----------
        x : np.array<np.float32>
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of features for each training example
            M should be equal to the value of n_input passed to the class constructor.
        y : np.array<np.float32>
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of outputs for each training example
            The value of N should be the same for both x and y.
            M should be equal to the value of n_output passed to the class constructor.
        learning_rate : float
            Learning rate to use
        n_epochs: int
            Number of epochs to train
        """
        # TODO your code here


# minimize this
def mse(y_pred, y_true):
    """
    y_pred : np.array<np.float32>
        The predicted outputs of a neural network.
        The dimension should be (N, M), where:
            - N is the number of training examples in the dataset
            - M is the number of outputs for each training example
    y_true : np.array<np.float32>
        The true labels.
        The dimension should be (N, M), where:
            - N is the number of training examples in the dataset
            - M is the number of outputs for each training example
    """
    assert y_pred.shape == y_true.shape, "Non-matching shape"
    diff = y_true - y_pred
    diff_squared = diff * diff
    total_loss = diff_squared.sum() / 2.0
    mean_loss = total_loss / y_true.shape[0]
    return mean_loss
