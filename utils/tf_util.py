import tensorflow as tf


def index_with_arrays(values, *index_arrays):
    """
    Select slices of an array based on aligned arrays for each dimension.
    Behaves equivalent to the numpy version:

    index_with_arrays(x, [1, 2, 3], [4, 5, 6])
     ==
    x[[1, 2, 3], [4, 5, 6]]  # in numpy


    index_with_arrays(x, [[1, 2], [3, 4]], [[5, 6], [6, 7]])
     ==
    x[[[1, 2], [3, 4]], [[5, 6], [6, 7]]]  # in numpy


    :param values: base values
    :param index_arrays: tf-Tensors of equal size. Indexing arrays correspond to
                         the dimensions in the values array.
                         Therefore, the number of indexing arrays cannot exceed the
                         tensor rank of values.
    :return: array with selections from values of shape
             indexing_arrays[any].shape + values.shape[len(indexing_arrays):]

    """
    for index_array in index_arrays:
        index_array.shape.assert_is_compatible_with(index_arrays[0].shape)

    expanded_indexing_arrays = [
        tf.expand_dims(index_array, -1)
        for index_array in index_arrays
    ]
    gather_indices = tf.concat(expanded_indexing_arrays, axis=-1, name='gather_indices')
    return tf.gather_nd(values, gather_indices)


def make_count_variable(name, init_value=0):
    return tf.get_variable(name, shape=(), dtype=tf.int32,
                           initializer=tf.constant_initializer(init_value))


def make_count_resetter(var):
    var_name = var.name.split(':')[0]
    return tf.assign(var, 0, name='reset_{}'.format(var_name))


def make_count_incrementer(var):
    var_name = var.name.split(':')[0]
    return tf.assign_add(var, 1, name='increment_{}'.format(var_name))


def logits_entropy(y_logit_batch):
    """
    Get the entropy for the probability distribution given by y_logit_batch whose
    last axis is interpreted as the logits of a probability distribution
    """
    powers = tf.exp(y_logit_batch)
    norm = tf.reduce_sum(powers, axis=-1, keepdims=True)
    entropies = (1 / norm) * tf.reduce_sum(powers * (tf.log(norm) - y_logit_batch),
                                           axis=-1, keepdims=True)
    return entropies

