import pytest
import tensorflow as tf
import numpy as np
from pytest import approx

from utils.math_util import softmax
from utils.tf_util import index_with_arrays, logits_entropy


class TestIndexWithArrays:
    def test_index_with_arrays(self):
        values = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        indices_1 = np.array([0, 2, 1, 0, 2])
        indices_2 = np.array([3, 2, 1, 2, 0])

        selection_desired = values[indices_1, indices_2]

        selection_actual = index_with_arrays(tf.constant(values),
                                             tf.constant(indices_1),
                                             tf.constant(indices_2))
        with tf.Session() as sess:
            _selection_test: np.ndarray = sess.run(selection_actual)

        assert _selection_test.tolist() == selection_desired.tolist()

    def test_index_with_arrays_multi_d(self):
        values = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        indices_1 = np.array([[[0, 2], [1, 0], [2, 2]]])
        indices_2 = np.array([[[3, 2], [1, 2], [0, 1]]])

        selection_desired = values[indices_1, indices_2]

        selection_actual = index_with_arrays(tf.constant(values),
                                             tf.constant(indices_1),
                                             tf.constant(indices_2))
        with tf.Session() as sess:
            _selection_test: np.ndarray = sess.run(selection_actual)

        assert _selection_test.tolist() == selection_desired.tolist()

    def test_index_with_arrays_out_of_bounds(self):
        values = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        indices_1 = np.array([0, 2, 1, 0, 3])  # 3 is out of bounds
        indices_2 = np.array([3, 2, 1, 2, 0])

        selection_actual = index_with_arrays(tf.constant(values),
                                             tf.constant(indices_1),
                                             tf.constant(indices_2))
        with pytest.raises(tf.errors.InvalidArgumentError):
            with tf.Session() as sess:
                _selection_test: np.ndarray = sess.run(selection_actual)

    def test_incompatible_shapes(self):
        values = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        indices_1 = np.array([0, 2, 1, 0])
        indices_2 = np.array([3, 2, 1, 2, 0])

        with pytest.raises(ValueError):
            index_with_arrays(tf.constant(values),
                              tf.constant(indices_1),
                              tf.constant(indices_2))


class TestLogitsEntropy:
    def run_test(self, y_logits, expected_entropies):
        y_logit_batch = tf.constant(y_logits)
        entropies = logits_entropy(y_logit_batch)
        with tf.Session() as sess:
            _entropies = sess.run(entropies)
        print('_entropies', _entropies)

        assert _entropies == approx(np.asarray(expected_entropies))

    def test_1d_even(self):
        y_logit_batch = [0.0, 0.0]
        _expected_entropies = np.log([2])
        self.run_test(y_logit_batch, _expected_entropies)

    def test_1d_even_2(self):
        y_logit_batch = [10.0, 10.0]
        _expected_entropies = np.log([2])
        self.run_test(y_logit_batch, _expected_entropies)

    def test_2d_evens(self):
        y_logit_batch = [[10.0, 10.0],
                         [0.0, 0.0],
                         [-5.0, -5.0]]
        _expected_entropies = [[np.log(2)],
                               [np.log(2)],
                               [np.log(2)]]
        self.run_test(y_logit_batch, _expected_entropies)

    def test_custom(self):
        rng = np.random.RandomState(12345)
        y_logit_batch = rng.uniform(-2, 2, size=(3, 4, 5))
        ps = softmax(y_logit_batch, axis=-1)
        entropies = -np.sum(ps * np.log(ps), axis=-1, keepdims=True)
        self.run_test(y_logit_batch, entropies)
