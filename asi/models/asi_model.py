"""
----------------------------
Interface specification
----------------------------
This interface should be a general interface for models that attempt to solve
these kinds of prediction problems.


"""
from typing import Dict

import numpy as np
import tensorflow as tf
from keras import Model, Input

from asi.temporal_matching import predict_match_interleaved
from asi.helpers.z_loss_fns import log_loss
from utils.tf_util import make_count_incrementer


def assert_delta_t_bounds_correctness(delta_t_bounds):
    if len(delta_t_bounds) != 2:
        raise ValueError('delta_t_bounds needs to have two entries')
    if delta_t_bounds[0] > delta_t_bounds[1]:
        raise ValueError('delta_t_bounds must specify the lower bound before the '
                         'upper bound.')
    if {type(delta_t_bounds[0]), type(delta_t_bounds[1])} != {int}:
        raise ValueError('Please provide integers as delta_t_bound')
    if delta_t_bounds[0] < 1:
        raise ValueError('The lower value delta_t_bounds must be at least 1')


def pad_trajectories(trajectory_list, pad_value):
    """
    :param trajectory_list: List of arrays of shape (trajectory_length,)+frame_shape
    :param pad_value: Value used to pad the output array
    :return: Padded array of shape
             (len(trajectory_list), max(trajectory_lengths))+frame_shape
    """
    trajectory_lengths = np.array([x_single.shape[0] for x_single in trajectory_list])

    def pad_x_single(x_single, max_length):
        assert x_single.shape[0] <= max_length
        pad_width_head = (0, max_length - x_single.shape[0])
        pad_width_tail = ((0, 0),) * (x_single.ndim - 1)
        return np.pad(x_single, (pad_width_head,) + pad_width_tail,
                      mode='constant', constant_values=pad_value)

    max_trajectory_length = max(trajectory_lengths)
    x_padded = np.array([pad_x_single(x_single, max_trajectory_length)
                         for x_single in trajectory_list])
    return x_padded, trajectory_lengths


class ASIModel:
    def __init__(self,
                 x_shape,
                 f_optimizer,
                 f: Model,
                 delta_t_bounds,
                 exploration_schedule,
                 schd_sampling_schedule,
                 train_step_counter: tf.Variable,
                 additional_metrics: Dict[str, tf.Tensor] = None,
                 x=None,
                 trajectory_lengths=None,
                 parallel_iterations=10,  # In while loop
                 z_loss_fn=log_loss,
                 forgiving=False
                 ):
        super().__init__()

        self.x_shape = x_shape

        self.f_optimizer = f_optimizer
        self.z_loss_fn = z_loss_fn

        assert_delta_t_bounds_correctness(delta_t_bounds)

        self.delta_t_bounds = delta_t_bounds
        self.parallel_iterations = parallel_iterations

        # Dimensions of x:
        #  0:   trajectory in batch
        #  1:   frame in trajectory
        #  2-n: dimensions within frame
        if x is None:
            self.x = Input(batch_shape=(None, None) + x_shape, name='x')
        else:
            self.x = Input(tensor=x)

        self.batch_size = tf.shape(self.x)[0]

        self.exploration_schedule = exploration_schedule
        self.exploration_p = tf.placeholder(tf.float32, shape=(), name='exploration_p')
        self.schd_sampling_schedule = schd_sampling_schedule
        self.schd_sampling_temperature = tf.placeholder(tf.float32, shape=(),
                                                        name='schd_sampling_temperature')

        self.forgiving = forgiving

        # Only used for the non-matching feedforward application
        self.n_predicted_frames = tf.placeholder(tf.int32, shape=(),
                                                 name='n_predicted_frames')

        # Since trajectories can have different lengths but they are processed
        # in a regular tensor fashion (zero-padded), we need to know where
        # they end in order to compute the correct losses
        if trajectory_lengths is None:
            self.trajectory_lengths = tf.placeholder(tf.int32,
                                                     shape=[None],
                                                     name='trajectory_lengths')
        else:
            self.trajectory_lengths = trajectory_lengths

        with tf.name_scope('Models'):
            self.f = f
            print('======== f summary ========')
            self.f.summary()

        if not additional_metrics:
            additional_metrics = dict()
        if type(additional_metrics) is not dict:
            raise ValueError('additional_metrics must be a dictionary!')
        self.additional_metrics = additional_metrics

        (self.z, self.z_shape, self.z_hat,
         self.z_loss, self.effective_z_hat_lengths,
         self.jump_timesteps, self.z_step_losses) = self.get_combined_model()

        self.z_hat_no_matching = self.get_no_matching_model()

        self.train_op = self.make_train_op()
        self._train_analytics = dict()

        self.train_step_counter = train_step_counter
        with tf.control_dependencies([self.train_op]):
            self.increment_train_step_counter = make_count_incrementer(
                self.train_step_counter)

    def train_on_trajectory_list(self, sess, x):
        x_padded, trajectory_lengths = self.prepare_trajectories(x)
        self.train_on_trajectories(sess, x_padded, trajectory_lengths)

    def train_on_trajectories(self, sess, x_padded=None, trajectory_lengths=None):
        metrics = self.analyze_batch_maybe_train(sess, execute_train_op=True,
                                                 x_padded=x_padded,
                                                 trajectory_lengths=trajectory_lengths,
                                                 fetch_effective_z_hat_lengths=True)
        self._train_analytics = metrics

    def analyze_batch(self, sess, x_padded=None, trajectory_lengths=None,
                      fetch_z=False, fetch_z_hat=False, fetch_jump_steps=False,
                      fetch_effective_z_hat_lengths=False, fetch_z_step_losses=False):
        metrics = self.analyze_batch_maybe_train(sess, execute_train_op=False,
                                                 x_padded=x_padded,
                                                 trajectory_lengths=trajectory_lengths,
                                                 fetch_z=fetch_z,
                                                 fetch_z_hat=fetch_z_hat,
                                                 fetch_jump_steps=fetch_jump_steps,
                                                 fetch_effective_z_hat_lengths=fetch_effective_z_hat_lengths,
                                                 fetch_z_step_losses=fetch_z_step_losses)
        return metrics

    def analyze_batch_maybe_train(self, sess, execute_train_op,
                                  x_padded=None, trajectory_lengths=None,
                                  fetch_z=False, fetch_z_hat=False,
                                  fetch_jump_steps=False,
                                  fetch_effective_z_hat_lengths=False,
                                  fetch_z_step_losses=False):
        """
        :param x_padded: padded array of trajectories to be fed into self.x (if needed)
        :param trajectory_lengths: vector of integer trajectory lengths
        :param sess: tensorflow session
        :param execute_train_op: Specifies whether parameter update should take place
                                 or whether the metrics just need to be evaluated
        :param fetch_z: Whether to return the directly embedded self.z
        :param fetch_z_hat: Whether to return the predicted embedded self.z_hat
        :param fetch_jump_steps: Whether to return the jump_timesteps
        :param fetch_effective_z_hat_lengths: Whether to return effective_z_hat_lengths
        :param fetch_z_step_losses: Whether to return z_step_losses
        :return: loss
        """
        if not ((x_padded is None) == (trajectory_lengths is None)):
            raise ValueError('x and trajectory_lengths have to either all be provided'
                             ' or none of them.')

        if x_padded is not None:
            assert trajectory_lengths is not None
            if np.shape(trajectory_lengths)[0] != np.shape(x_padded)[0]:
                raise ValueError('Trajectory lengths have to correspond to first '
                                 'dimension of x-batch.')

            feed_dict = {
                self.x: x_padded,
                self.trajectory_lengths: trajectory_lengths
            }
        else:
            feed_dict = dict()

        run_dict = {
            'z_loss': self.z_loss,
            'effective_z_hat_lengths': self.effective_z_hat_lengths,
            'trajectory_lengths': self.trajectory_lengths,
        }

        run_dict.update(self.additional_metrics)

        if execute_train_op:
            current_train_step_count = sess.run(self.train_step_counter)
            run_dict['train_op'] = self.train_op
            run_dict['increment_train_step_counter'] = self.increment_train_step_counter
            exploration_p_val = self.exploration_schedule(current_train_step_count)
            schd_sampling_temperature_val = self.schd_sampling_schedule(
                current_train_step_count)
        else:
            exploration_p_val = 0.0
            schd_sampling_temperature_val = 0.0

        feed_dict.update({
            self.exploration_p: exploration_p_val,
            self.schd_sampling_temperature: schd_sampling_temperature_val,
        })

        metrics_keys = [
            'z_loss',
        ]

        metrics_keys.extend(self.additional_metrics.keys())

        if fetch_effective_z_hat_lengths:
            metrics_keys.append('effective_z_hat_lengths')

        if fetch_z:
            run_dict['z'] = self.z
            metrics_keys.append('z')

        if fetch_z_hat:
            run_dict['z_hat'] = self.z_hat
            metrics_keys.append('z_hat')

        if fetch_jump_steps:
            run_dict['jump_timesteps'] = self.jump_timesteps
            metrics_keys.append('jump_timesteps')

        if fetch_z_step_losses:
            run_dict['z_step_losses'] = self.z_step_losses
            metrics_keys.append('z_step_losses')

        run_output = sess.run(run_dict,
                              feed_dict)

        additional_metrics = {
            'schd_sampling_temperature': schd_sampling_temperature_val,

            'exploration_p': exploration_p_val
        }

        if 'effective_z_hat_lengths' in run_output:
            run_output['effective_z_hat_lengths'] = np.mean(
                run_output['effective_z_hat_lengths'])

        metrics = {**{key: run_output[key] for key in metrics_keys},
                   **additional_metrics}
        return metrics

    def prepare_trajectories(self, x_list):
        for x_single in x_list:
            if x_single.shape[1:] != self.x_shape:
                raise ValueError('A trajectory must consist of frames with the '
                                 'correct observation shape ({}), '
                                 'got a trajectory of shape {}'.format(self.x_shape,
                                                                       x_single.shape))

        x_padded, trajectory_lengths = pad_trajectories(x_list, 0.0)
        return x_padded, trajectory_lengths

    def predict_n_steps(self, x_1, n, sess):
        """
        :param x_1: Batch of first frames
        :param n: Number of steps to predict
        :param sess: TensorFlow session
        """

        expected_x_1_shape = tuple(self.x.shape[2:].as_list())
        if x_1.shape[1:] != expected_x_1_shape:
            raise ValueError('Expected shape (?,)+{}, but got x_1 with '
                             'shape {}'.format(expected_x_1_shape, x_1.shape))

        _feed_dict = {
            self.x: x_1[:, np.newaxis],
            self.n_predicted_frames: n,
        }
        _z_hat = sess.run(self.z_hat_no_matching,
                          _feed_dict)
        return _z_hat

    def get_no_matching_model(self):
        loop_i = tf.constant(0, dtype=tf.int32)

        # Transpose first two dimensions of z, such that the axes correspond to:
        #  0:   frame within trajectory
        #  1:   trajectory within batch
        #  2-n: axes of individual frame
        transpose_order = [1, 0] + list(range(2, 2 + len(self.z_shape)))

        z_no_matching = self.x

        z_transposed = tf.transpose(z_no_matching,
                                    perm=transpose_order)

        z_pred_accumulator = tf.zeros(
            # Transposed - therefore the trajectory is on axis 0, batch is on axis 1
            [self.n_predicted_frames, self.batch_size] + self.z_shape,
            dtype=tf.float32,
            name='z_pred_accumulator'
        )

        def loop_condition(i, all_obs):
            return tf.less(i, self.n_predicted_frames)

        def loop_body(i, z_transposed_acc):
            prediction = tf.cond(
                tf.equal(i, 0),
                # At first, predict on the initial latent state
                true_fn=lambda: self.f(z_transposed[0]),
                # Afterwards, predict recursively on previous prediction
                false_fn=lambda: self.f(z_transposed_acc[i - 1])
            )

            # e.g.: 3 -> [[3]]
            # First axis is for the different values we may want to insert
            # (but we only insert one, since we transposed before)
            indices = i[tf.newaxis, tf.newaxis]

            # Main idea: start with zeros, and add the next prediction as a
            # padded array, such that it only affects the current index.
            # Could also have been solved with concatenation, but maybe
            # this could be faster, since we already know the eventual shape?
            # (On the other hand, no idea how fast tf.scatter_nd is)
            all_obs_diff = tf.scatter_nd(indices=indices,
                                         updates=tf.expand_dims(prediction, axis=0),
                                         shape=tf.shape(z_pred_accumulator))
            updated_all_obs = tf.add(z_transposed_acc, all_obs_diff)
            return tf.add(i, 1), updated_all_obs

        _, z_hat_transposed = tf.while_loop(loop_condition, loop_body,
                                            loop_vars=[loop_i, z_pred_accumulator],
                                            parallel_iterations=self.parallel_iterations)

        z_hat_no_matching = tf.transpose(z_hat_transposed,
                                         perm=transpose_order)
        return z_hat_no_matching

    def get_combined_model(self):
        z = self.x
        z_shape = z.shape[2:].as_list()

        match_result = predict_match_interleaved(z,
                                                 self.f,
                                                 self.delta_t_bounds,
                                                 self.z_loss_fn,
                                                 self.trajectory_lengths,
                                                 self.exploration_p,
                                                 self.schd_sampling_temperature)
        (mean_z_loss, effective_z_hat_lengths,
         z_hat, jump_timesteps, z_step_losses) = match_result

        return (z, z_shape, z_hat,
                mean_z_loss, effective_z_hat_lengths, jump_timesteps, z_step_losses)

    def make_train_op(self):
        f_weights = self.f.trainable_weights
        if f_weights:
            train_f = self.f_optimizer.minimize(self.z_loss,
                                                var_list=f_weights,
                                                name='train_f')
        elif self.forgiving:
            train_f = tf.no_op(name='train_f_noop')
        else:
            raise ValueError('No f-weights found.')

        train_op = train_f
        return train_op

    @property
    def train_analytics(self):
        return self._train_analytics
