import numpy as np
import tensorflow as tf
from inspect import getmembers, isfunction
from asi.helpers.z_loss_fns import symmetric_log_loss, log_loss
import asi.helpers.z_loss_fns


def test_general_compatibility():
    """
    For all functions in the z-loss-function module, check if they
    satisfy the requirement of broadcastability.
    """
    rng = np.random.RandomState(1234)

    z_shape = (3, 5, 7)
    target = rng.uniform(0, 1,
                         (17, 19) + z_shape)
    prediction = rng.uniform(0, 1,
                             (17, 1,) + z_shape)
    loss_fns = [x[1] for x in getmembers(asi.helpers.z_loss_fns)
                if isfunction(x[1])]

    print('loss_fns', loss_fns)
    for loss_fn in loss_fns:
        losses = loss_fn(tf.constant(target),
                         tf.constant(prediction))
        with tf.Session().as_default():
            _losses = losses.eval()

        assert _losses.shape == target.shape


def test_symmetric_log_loss():
    target = tf.placeholder(tf.float32, name='target')
    prediction = tf.placeholder(tf.float32, name='prediction')

    losses = symmetric_log_loss(target, prediction)

    log_losses = log_loss(target, prediction)
    reverse_log_losses = log_loss(prediction, target)

    with tf.Session().as_default():
        # Symmetry
        _loss_a = losses.eval({target: 0.9, prediction: 0.2})
        _loss_b = losses.eval({target: 0.2, prediction: 0.9})
        assert _loss_a == _loss_b

        # Identical to summed up asymmetric loss
        asymmetric_sum = log_losses + reverse_log_losses
        _loss_c = asymmetric_sum.eval({target: 0.9, prediction: 0.2})
        assert np.isclose(_loss_a, _loss_c)


