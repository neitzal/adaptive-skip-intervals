import tensorflow as tf


def symmetric_log_loss(a, b):
    eps = 1e-6
    loss = tf.add(-a * tf.log(b + eps) -
                  (1 - a) * tf.log(1 - b + eps),
                  -b * tf.log(a + eps) -
                  (1 - b) * tf.log(1 - a + eps),
                  name='symmetric_log_loss_{}_{}'.format(a.name.split(':')[0],
                                                         b.name.split(':')[0]))
    return loss


def log_loss(ref_y, predicted_prob):
    """
    Broadcastable version of cross-entropy loss
    """
    eps = 1e-6
    loss = tf.add(-ref_y * tf.log(predicted_prob + eps),
                  -(1 - ref_y) * tf.log(1 - predicted_prob + eps),
                  name='log_loss_{}_{}'.format(ref_y.name.split(':')[0],
                                               predicted_prob.name.split(':')[0]))
    return loss


def squared_error(targets, predictions):
    return (targets - predictions) ** 2
