import numpy as np


def sigmoid(x):
    """
    Numerically-stable sigmoid function.
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


def log_loss(integer_labels, predictions, epsilon=1e-6):
    """
    Take log loss by interpreting the last axis of predictions as probabilities
    for the integer_labels.
    """
    if np.ndim(predictions) != 2:
        raise ValueError('Currently, only 2D inputs are supported')
    if np.shape(integer_labels) != np.shape(predictions)[:-1]:
        raise ValueError(
            'Incompatible shapes: integer_labels.shape={}, '
            'predictions.shape={}'.format(
                np.shape(integer_labels), np.shape(predictions)))

    n = np.shape(predictions)[-1]
    one_hot_labels = np.eye(n, n)[integer_labels]

    return -np.log(np.sum(one_hot_labels * predictions, axis=-1) + epsilon)


def cross_entropy_loss_bc(ground_truth_p, predicted_p, epsilon=1e-6):
    """
    For broadcast-compatible input-arrays
    """
    ground_truth_p = np.asarray(ground_truth_p)
    predicted_p = np.asarray(predicted_p)
    losses = (-ground_truth_p * np.log(predicted_p + epsilon)
              - (1 - ground_truth_p) * np.log(1 - predicted_p + epsilon))
    return losses
