import logging

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Lambda

from asi.helpers import annealing_schedules, z_loss_fns
from utils.tf_util import make_count_variable

logging.getLogger('tensorflow').disabled = True

from asi.models.asi_model import ASIModel, pad_trajectories
from tests.example_models import make_example_f, make_constant_keras_model

rng = np.random.RandomState(1234)


def get_batch_loss(predictor):
    """
    Legacy batch loss
    """
    return predictor.train_analytics['z_loss']


def test__make_constant_keras_model():
    input_shape = (2, 4)
    model = make_constant_keras_model(input_shape=input_shape, output_tensor=[42.0])
    x = np.random.uniform(0, 1, (3,) + input_shape)
    y = model.predict_on_batch(x)
    assert y.tolist() == [[42.], [42.], [42.]]

    input_shape = (1, 1, 1, 1, 1, 1, 1, 1)
    model = make_constant_keras_model(input_shape=input_shape, output_tensor=43.0)
    x = np.random.uniform(0, 1, (5,) + input_shape)
    y = model.predict_on_batch(x)
    assert y.tolist() == [43., 43., 43., 43., 43.]

    input_shape = (7,)
    model = make_constant_keras_model(input_shape=input_shape,
                                      output_tensor=[44.0, 32.0])
    x = np.random.uniform(0, 1, (3,) + input_shape)
    y = model.predict_on_batch(x)
    assert y.tolist() == [[44., 32.], [44., 32.], [44., 32]]


def _make_keras_mean_model(input_shape, bias=0.):
    inpt = Input(shape=input_shape)

    def get_mean(x):
        mean = tf.reduce_mean(x, axis=tuple(range(-1, -len(input_shape) - 1, -1)))
        return tf.expand_dims(mean + bias, axis=-1)

    output = Lambda(get_mean)(inpt)
    return Model(inpt, output)


class TestBatchIterativePredictor:

    def test_pad_trajectories(self):
        trajectory_list = [
            np.array([[1], [2], [3]]),
            np.array([[4]]),
            np.array([[5], [6], [7], [8]]),
            np.array([[9], [10], [11]]),
        ]
        padded_trajectories, trajectory_lengths = pad_trajectories(trajectory_list,
                                                                   pad_value=42)

        # @formatter:off
        expected_output = [
            [[ 1], [ 2], [ 3], [42]],
            [[ 4], [42], [42], [42]],
            [[ 5], [ 6], [ 7], [ 8]],
            [[ 9], [10], [11], [42]],
        ]
        # @formatter:on
        assert padded_trajectories.tolist() == expected_output

    def test_predict_no_errors(self):
        """
        Predictor with delta_t_bounds (1, 1) - should be equivalent to the version
        without dynamic time matching.
        Assertions to ensure that behavior stays the same (these are just the outputs
        of the predictor from 2018-03-22.
        """
        tf.reset_default_graph()
        rng = np.random.RandomState(1234)
        x_shape = (5, 3)
        x_1 = rng.uniform(0, 1, (14,) + x_shape)

        train_step_counter = make_count_variable('train_step_counter', 0)
        predictor = ASIModel(x_shape,
                             f_optimizer=tf.train.AdamOptimizer(),
                             f=make_example_f(list(x_shape)),
                             delta_t_bounds=(1, 1),
                             exploration_schedule=None,  # unused
                             schd_sampling_schedule=None,  # unused
                             parallel_iterations=10,
                             train_step_counter=train_step_counter,
                             )
        init_vars = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_vars)
            z_hat = predictor.predict_n_steps(x_1, n=4, sess=sess)

        print('z_hat.shape', z_hat.shape)
        assert z_hat.shape == (14, 4, 5, 3)

        print('z_hat.mean()', z_hat.mean())
        assert np.isclose(z_hat.mean(), 0.4999843)

    def test_train_multiple_trajectories_no_errors(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(1234)

        x_shape = (2, 3)

        trajectory_lengths = [4, 3, 5, 2]
        x_list = [rng.uniform(0, 1, (trajectory_length,) + x_shape)
                  for trajectory_length in trajectory_lengths]

        train_step_counter = make_count_variable('train_step_counter', 0)
        predictor = ASIModel(x_shape,
                             f_optimizer=tf.train.AdamOptimizer(),
                             f=make_example_f(list(x_shape)),
                             delta_t_bounds=(1, 1),
                             exploration_schedule=annealing_schedules.constant_zero,
                             schd_sampling_schedule=annealing_schedules.constant_zero,
                             parallel_iterations=10,
                             train_step_counter=train_step_counter,
                             z_loss_fn=z_loss_fns.symmetric_log_loss
                             )
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)
            predictor.train_on_trajectory_list(sess, x_list)
            _loss = get_batch_loss(predictor)
            print('_loss 1', _loss)
            assert np.isclose(_loss, 1.7272503)

            predictor.train_on_trajectory_list(sess, x_list)
            _loss = get_batch_loss(predictor)
            print('_loss 2', _loss)
            assert np.isclose(_loss, 1.7265959)

    def test_predict_simple_models(self):
        tf.reset_default_graph()
        x_shape = (2, 3)
        z_shape = x_shape
        n_timesteps = 4

        # f adds one
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: x + 1)(z_1), name='f')

        train_step_counter = make_count_variable('train_step_counter', 0)
        predictor = ASIModel(x_shape,
                             f_optimizer=tf.train.AdamOptimizer(),
                             f=f,
                             delta_t_bounds=(1, 1),
                             exploration_schedule=annealing_schedules.constant_zero,
                             schd_sampling_schedule=annealing_schedules.constant_zero,
                             parallel_iterations=10,
                             train_step_counter=train_step_counter,
                             forgiving=True)

        x_bias = 10.0
        x_1 = np.stack((np.ones(x_shape, dtype=np.float32),
                        x_bias + np.ones(x_shape, dtype=np.float32)),
                       axis=0)
        print('x_1.shape', x_1.shape)
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)
            _z_hat = predictor.predict_n_steps(x_1, n=n_timesteps,
                                               sess=sess)
            desired_z_hat_a = (np.arange(2, 2 + n_timesteps)[:, None, None] +
                               np.zeros((n_timesteps,) + x_shape))
            desired_z_hat_b = (x_bias + np.arange(2, 2 + n_timesteps)[:, None, None] +
                               np.zeros((n_timesteps,) + x_shape))

            assert _z_hat.tolist() == [desired_z_hat_a.tolist(),
                                       desired_z_hat_b.tolist()]

    def test_train_f(self):
        """
        The real dynamics is: x -> x+1
        Parametrize model f with one parameter theta, to encode the function x -> x + theta
        Check if the loss can be reduced
        """
        tf.reset_default_graph()
        x_shape = (1,)
        z_shape = x_shape
        n_timesteps = 5

        # f adds theta
        z_1 = Input(shape=z_shape)
        with tf.variable_scope('test_train_f'):
            theta = tf.get_variable('theta_2', shape=(), dtype=tf.float32,
                                    initializer=tf.initializers.constant(0.0))
        f_layer = Lambda(lambda x: x + theta, trainable=True, )
        f_layer.trainable_weights = [theta]
        f = Model(z_1, f_layer(z_1), name='f')
        print('f.trainable_weights', f.trainable_weights)

        learning_rate = 0.01
        train_step_counter = make_count_variable('train_step_counter', 0)
        predictor = ASIModel(x_shape,
                             f=f,
                             delta_t_bounds=(1, 1),
                             exploration_schedule=annealing_schedules.constant_zero,
                             schd_sampling_schedule=annealing_schedules.constant_zero,
                             parallel_iterations=10,
                             train_step_counter=train_step_counter,
                             z_loss_fn=lambda a, b: tf.square(a - b),
                             f_optimizer=tf.train.GradientDescentOptimizer(
                                learning_rate),
                             forgiving=True
                             )

        x_observed = (np.zeros((n_timesteps,) + x_shape, dtype=np.float32) +
                      np.arange(0, n_timesteps).reshape(
                          (1, -1) + tuple([1] * len(x_shape))))
        x_1 = x_observed[:, 0]
        batch_size = x_observed.shape[0]

        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)
            _z_hat = predictor.predict_n_steps(x_1, n=n_timesteps,
                                                               sess=sess)
            print('_z_hat:', _z_hat)
            desired_z_hat = np.zeros((batch_size, n_timesteps, 1))
            assert _z_hat.tolist() == desired_z_hat.tolist()

            predictor.train_on_trajectories(sess, x_observed,
                                            trajectory_lengths=[n_timesteps])
            assert predictor.train_analytics['z_loss'] == 7.5

            print('predictor.train_analytics', predictor.train_analytics)

            new_theta = sess.run(theta)
            print('new_theta', new_theta)
            assert np.isclose(new_theta, 0.15)

