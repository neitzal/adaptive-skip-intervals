import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Lambda
from pytest import approx

from asi.helpers import z_loss_fns
from asi.temporal_matching import select_frame_slices, \
    mix_with_random_values, reduce_losses, cut_values, predict_match_interleaved, \
    accumulate_in_loop


def _replace_ellipses(prediction, ground_truth):
    if prediction is ...:
        try:
            return list(ground_truth)
        except TypeError:
            return ground_truth
    else:
        try:
            return [_replace_ellipses(x, y) for x, y in zip(prediction, ground_truth)]
        except TypeError:
            return prediction


def test_replace_ellipses():
    a = [[[1, ...], [..., 4], ...],
         ...,
         [..., [15, 16], [..., 18]]]

    b = [[[1, 2], [3, 4], [5, 6]],
         [[7, 8], [9, 10], [11, 12]],
         [[13, 14], [15, 16], [17, 18]]]

    a_augmented = _replace_ellipses(a, b)
    assert a_augmented == b

    a[0][0][0] = 2
    a_augmented = _replace_ellipses(a, b)
    assert a_augmented != b


class TestSelectFrameSlices():
    def test_regular(self):
        _i_actual = [1, 0, 3, 2]
        delta_t_bounds = (2, 4)
        # frames = np.arange(4*5).reshape(4, 5)

        # @formatter:off
        frames = [[ 0,  1,  2,  3,  4],
                  [ 5,  6,  7,  8,  9],
                  [10, 11, 12, 13, 14],
                  [15, 16, 17, 18, 19]]
        # @formatter:on

        selected_frames = select_frame_slices(tf.constant(_i_actual),
                                              tf.constant(frames),
                                              delta_t_bounds)

        with tf.Session().as_default():
            _selected_frames: np.ndarray = selected_frames.eval()

        assert _selected_frames.tolist() == [[3, 4, 4],
                                             [7, 8, 9],
                                             [14, 14, 14],
                                             [19, 19, 19]]

    def test_out_of_bounds(self):
        """
        Desired behavior: repeat the last frame
        """
        _i_actual = [4, 4, 4]
        delta_t_bounds = (0, 2)
        # frames = np.arange(4*5).reshape(4, 5)

        # @formatter:off
        frames = [[ 0,  1,  2,  3,  4],
                  [ 5,  6,  7,  8,  9],
                  [10, 11, 12, 13, 14]]
        # @formatter:on

        selected_frames = select_frame_slices(tf.constant(_i_actual),
                                              tf.constant(frames),
                                              delta_t_bounds)

        with tf.Session().as_default():
            _selected_frames: np.ndarray = selected_frames.eval()

        assert _selected_frames.tolist() == [[4, 4, 4],
                                             [9, 9, 9],
                                             [14, 14, 14]]


def test_mix_with_random_values():
    # First, determine the random pattern
    random_seed = 23456
    random_determiners = tf.random_uniform((5,),
                                           0, 1.0,
                                           dtype=tf.float32,
                                           seed=random_seed,
                                           name='random_determiners')
    random_selection = tf.random_uniform((5,), 0, 8,
                                         seed=random_seed + 345,
                                         dtype=tf.int32, name='random_skip')
    with tf.Session().as_default():
        selection = random_determiners.eval() < 0.5
        random_indices = random_selection.eval()

        print('random_determiners.eval() < 0.5', selection)
        print('random_selection.eval()', random_indices)

    assert selection.tolist() == [True, False, True, True, False]
    assert random_indices.tolist() == [4, 6, 0, 2, 5]

    # Now that we know the random pattern, we know what to expect:

    original_indices = tf.constant([0, 1, 2, 3, 4])
    expected_mixed_indices = [4, 1, 0, 2, 4]

    batch_size = tf.shape(original_indices)[0]
    actual_mixed_indices = mix_with_random_values(original_indices,
                                                  max_index=8,
                                                  exploration_p=0.5,
                                                  seed=random_seed,
                                                  batch_size=batch_size)
    with tf.Session().as_default():
        _actual_mixed_indices = actual_mixed_indices.eval()

    assert _actual_mixed_indices.tolist() == expected_mixed_indices


def test_reduce_losses():
    loss_scalars = [
        [
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ],
        [
            [[11., 12.], [13., 14]],
            [[15., 16.], [17., 18.]]
        ],
    ]
    reduction_ndim = 2
    reduced_losses = reduce_losses(loss_scalars=tf.constant(loss_scalars),
                                   reduction_ndim=reduction_ndim)
    with tf.Session().as_default():
        _reduced_losses = reduced_losses.eval()

    assert _reduced_losses.tolist() == [[2.5, 6.5], [12.5, 16.5]]


def test_cut_values():
    # @formatter:off
    values =     [ 1.0,  2.0,   3.0,  4.0,   5.0]
    keep_array = [True, True, False, True, False]
    # @formatter:on

    kept_count, value_sum = cut_values(tf.constant(values, tf.float32),
                                       tf.constant(keep_array, tf.bool))

    with tf.Session() as sess:
        _kept_count, _value_sum = sess.run([kept_count, value_sum])

    assert _kept_count == 3
    assert _value_sum == 7.0


class TestAccumulateInLoop:
    def test_1d(self):
        acc = tf.placeholder(tf.float32, shape=(None,))
        index = tf.placeholder(tf.int32, shape=())
        value = tf.placeholder(tf.float32, shape=())
        result = accumulate_in_loop(acc, index, value)

        _acc = [1., 2., 3., 4.]
        _index = 2
        _value = 10.
        _expected_result = [1., 2., 13., 4.]

        with tf.Session() as sess:
            _result = sess.run(result,
                               feed_dict={acc: _acc, index: _index, value: _value})
        assert _result.tolist() == _expected_result

    def test_2d(self):
        acc = tf.placeholder(tf.float32, shape=(None, None))
        index = tf.placeholder(tf.int32, shape=())
        value = tf.placeholder(tf.float32, shape=(None,))
        result = accumulate_in_loop(acc, index, value)

        _acc = [[1., 2.], [3., 4.], [5., 6.]]
        _index = 1
        _value = [10., 100.]
        _expected_result = [[1., 2.], [13., 104.], [5., 6.]]

        with tf.Session() as sess:
            _result = sess.run(result,
                               feed_dict={acc: _acc, index: _index, value: _value})
        assert _result.tolist() == _expected_result


class TestPredictAndMatchInterleaved:
    @staticmethod
    def run_test(*args, **kwargs):
        """Dispatches arguments to the test, once with gpu, once with cpu"""
        TestPredictAndMatchInterleaved.actually_run_test(*args, **kwargs)
        with tf.device('/cpu:0'):
            TestPredictAndMatchInterleaved.actually_run_test(*args, **kwargs)

    @staticmethod
    def actually_run_test(z_shape,
                          ground_truth_trajectories,
                          f, delta_t_bounds,
                          _sched_sampling_temperature,
                          _exploration_p,
                          expected_mean_loss,
                          expected_effective_z_hat_lengths,
                          expected_jump_timesteps,
                          expected_z_step_losses=None):
        # Tensors
        z = tf.placeholder(tf.float32, (None, None) + z_shape, name='z')
        trajectory_lengths = tf.placeholder(tf.int32, shape=(None,),
                                            name='trajectory_lengths')
        exploration_p = tf.placeholder(tf.float32, shape=(),
                                       name='exploration_p')
        sched_sampling_temperature = tf.placeholder(tf.float32, shape=(),
                                                    name='sched_sampling_temperature')

        _trajectory_lengths = [len(t) for t in ground_truth_trajectories]

        _z = np.zeros(
            (len(ground_truth_trajectories), max(_trajectory_lengths)) + z_shape,
            np.float32)

        for i, t in enumerate(ground_truth_trajectories):
            _z[i, :len(t)] = np.array(t).reshape((len(t),) + (1,) * len(z_shape))

        match_result = predict_match_interleaved(z, f, delta_t_bounds,
                                                 z_loss_fns.squared_error,
                                                 trajectory_lengths, exploration_p,
                                                 sched_sampling_temperature,
                                                 random_seed=123)
        (mean_z_loss, effective_z_hat_lengths, z_hat,
         jump_timesteps, z_step_losses) = match_result

        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)
            (_mean_z_loss,
             _effective_z_hat_lengths,
             _z_hat,
             _jump_timesteps,
             _z_step_losses) = sess.run(
                [mean_z_loss, effective_z_hat_lengths, z_hat,
                 jump_timesteps, z_step_losses],
                {z: _z,
                 trajectory_lengths: _trajectory_lengths,
                 exploration_p: _exploration_p,
                 sched_sampling_temperature: _sched_sampling_temperature})
            assert _z_hat.shape[1] == np.max(_effective_z_hat_lengths)
            assert expected_effective_z_hat_lengths == _effective_z_hat_lengths.tolist()
            assert np.isclose(expected_mean_loss, _mean_z_loss)

            print('_jump_timesteps.shape', _jump_timesteps.shape)
            print('_jump_timesteps', _jump_timesteps)

            assert expected_jump_timesteps == _jump_timesteps.tolist()
            # print('_mean_z_loss', _mean_z_loss)
            print('_effective_z_hat_lengths', _effective_z_hat_lengths)
            # print('_z_hat:\n', _z_hat)

            # print('_z_step_losses', _z_step_losses)

            if expected_z_step_losses:
                expected_z_step_losses = _replace_ellipses(expected_z_step_losses,
                                                           _z_step_losses)
                assert np.array(expected_z_step_losses) == approx(_z_step_losses)

    def test_onestep(self):
        z_shape = (2,)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 2.0,  4.0,  8.0, 16.0, 32.0],
            [ 5.0, 10.0, 20.0, 40.0],
            [ 6.0, 12.0, 24.0, 48.0, 96.0],
            [ 0.5,  1.0,  2.0,  4.0,  8.0, 16.0],
            [ 1.0, 2.0]
        ]
        # @formatter:on

        delta_t_bounds = (1, 2)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.0

        expected_effective_z_hat_lengths = [4, 3, 4, 5, 1]

        expected_jump_timesteps = [
            [1, 2, 3, 4, 4],
            [1, 2, 3, 3, 3],
            [1, 2, 3, 4, 4],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1]
        ]

        # @formatter:off
        # ... == "don't care"
        # The losses which reach above the true trajectory are computed assuming
        # zero padded true trajectories
        #
        expected_z_step_losses = [
            [[0.0, 16.0], [0.0, 64.0], [0.0, 256.], [0.0, 1024.0], ...],
            [[0.0, 100.0], [0.0, 400.0], [0.0, 1600.0], ..., ...],
            [[(12. - 12) ** 2, (24. - 12) ** 2],
             [(24. - 24) ** 2, (48. - 24) ** 2],
             [(48. - 48) ** 2, (96. - 48) ** 2],
             [(96. - 96) ** 2, (0. - 96) ** 2],
             ...,],
            [[0.0, 1.0],
             [0.0, 4.0],
             [0.0, 16.0],
             [0.0, 64.0],
             [0.0, 0.0],], # here, the loss is zero because the longest trajectory
                          # isn't zero padded but the last frame is repeated! Tricky.
            [[0.0, 4.0], ..., ..., ..., ..., ]
        ]
        # @formatter:on

        expected_mean_loss = 0.0
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps,
                      expected_z_step_losses)

    def test_multistep(self):
        z_shape = (7, 13)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 2.0,  3.0,  4.0,  8.0, 10.0, 16.0],  # Effective steps: 3
            [ 5.0, 10.0, 12.0, 15.0, 20.0, 40.0],  # Effective steps: 3
            [ 6.0, 12.0, 24.0, 48.0, 96.0],        # Effective steps: 4
            [ 0.5, 0.7, 0.9, 1.0, 1.9, 1.96, 2.0, 3.5, 4.0],  # Effective steps: 3
            [ 1.0, 2.0]                            # Effective steps: 1
        ]
        # @formatter:on

        delta_t_bounds = (1, 3)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.0

        expected_effective_z_hat_lengths = [3, 3, 4, 3, 1]

        expected_jump_timesteps = [
            [2, 3, 5, 5],
            [1, 4, 5, 5],
            [1, 2, 3, 4],
            [3, 6, 8, 8],
            [1, 1, 1, 1],
        ]

        expected_mean_loss = 0.0
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)

    def test_nonzero_loss_due_to_insufficient_stepsize(self):
        z_shape = (2,)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 4.0,  5.0,  7.0,  8.0, 15.0, 16.0, 31.0],  # Prediction: 8.0, 16.0, 32.0
            [ 5.0,  6.0,  9.0, 10.1, 21.0],              # Prediction: 10.0, 20.0
        ]
        # @formatter:on

        expected_jump_timesteps = [
            [2, 4, 6],
            [2, 4, 4]
        ]

        delta_t_bounds = (1, 2)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.0

        expected_effective_z_hat_lengths = [3, 2]

        expected_mean_loss = 1.0
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)

    def test_matching_exploration_1(self):
        z_shape = (2,)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 5.0, 10.0, 20.0, 40.0],
            [ 2.0,  4.0,  8.0, 16.0, 32.0, 42.0, 64.0],
            [ 1.0, 2.0]
        ]
        # @formatter:on

        delta_t_bounds = (1, 2)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.5

        # Compute expected values
        random_determiners = [[0.04080963, 0.20842123, 0.09180295],
                              [0.11709571, 0.00243473, 0.64621484],
                              [0.18114924, 0.31590247, 0.8326131],
                              [0.2236327, 0.21146584, 0.9371401]]

        # Skip-values are delta_t_bounds[0] plus this:
        random_selections = [[1, 1, 0],
                             [1, 0, 1],
                             [1, 0, 1],
                             [0, 1, 1]]
        # @formatter:off
        # Explanation for the second term in the first column:
        # Even though we are supposed to jump by 2, we jump by
        # one because otherwise we would be outside of the trajectory
        error_terms = [(20.0 - 10.0), ( 8.0 -  4.0), ( 2.0 -  2.0),
                       (40.0 - 20.0), (16.0 -  8.0),
                                      (32.0 - 16.0),
                                      (64.0 - 32.0)
                       ]
        # @formatter:on

        expected_effective_z_hat_lengths = [2, 4, 1]

        expected_jump_timesteps = [
            [2, 3, 3, 3],
            [2, 3, 4, 6],
            [1, 1, 1, 1]
        ]

        squared_errors = np.square(error_terms)
        expected_mean_loss = np.mean(squared_errors)
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)

    def test_matching_exploration_2(self):
        z_shape = (2,)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 5.0, 10.0, 15.0, 20.0, 25.0, 40.0],
            [ 2.0,  4.0,  8.0, 16.0, 32.0, 42.0, 64.0],
        ]
        # @formatter:on

        delta_t_bounds = (1, 3)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.5

        # Compute expected values
        random_determiners = [[0.04080963, 0.20842123],
                              [0.700411677, 0.173114538],
                              [0.510741353, 0.819653034],
                              [0.181149244, 0.315902472]]

        # Skip-values are delta_t_bounds[0] plus this:
        random_selections = [[1, 2],
                             [0, 0],
                             [0, 1],
                             [1, 2]]
        # @formatter:off
        # Explanation for the fourth term in the second column:
        # Even though we are supposed to jump by 3, we jump by
        # one because otherwise we would be outside of the trajectory
        error_terms = [
                        (15.0 - 10.0),  (16.0 - 4.0),
                        (20.0 - 20.0),  (32.0 - 8.0),
                        (40.0 - 40.0),  (42.0 - 16.0),
                                        (64.0 - 32.0)
                      ]
        # @formatter:on

        expected_effective_z_hat_lengths = [3, 4]

        expected_jump_timesteps = [
            [2, 3, 5, 5],
            [3, 4, 5, 6]
        ]

        squared_errors = np.square(error_terms)
        expected_mean_loss = np.mean(squared_errors)
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)

    def test_scheduled_sampling(self):
        z_shape = (2,)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 2.0,  5.0, 11.0, 23.0, 47.0],
            [ 5.0,  9.0, 17.0, 33.0]
        ]
        # @formatter:on

        # All scheduled_sampling_randoms:
        # (TEMPERATURE: 0.8)
        # ~~ [0.861665606 0.681721091]
        # ~~ [0.898319125 0.0674775839]
        # ~~ [0.626882434 0.732009053]
        # ~~ [0.856670499 0.38668]
        #
        # -> Predictions
        # (2.0) ->  4.0   (4.0) ->  8.0  (11.0) -> 22.0  (22.0) -> 44.0
        # (5.0) -> 10.0   (9.0) -> 18.0  (17.0) -> 34.0

        # @formatter:off
        losses = [( 5.0 - 4.0),  (10.0 - 9.0),
                  (11.0 - 8.0),  (17.0 - 18.0),
                  (23.0 - 22.0), (33.0 - 34.0),
                  (47.0 - 44.0)
                  ]
        # @formatter:on

        delta_t_bounds = (1, 2)
        _sched_sampling_temperature = 0.8
        _exploration_p = 0.0

        expected_effective_z_hat_lengths = [4, 3]

        expected_jump_timesteps = [
            [1, 2, 3, 4],
            [1, 2, 3, 3]
        ]

        expected_mean_loss = np.mean(np.square(losses))
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)

    def test_longer_trajectory_but_fewer_effective_steps(self):
        """
        The shorter trajectory will take more f-steps than the longer one,
        since it makes greater strides.
        This can work on a GPU because it does not check array boundaries.
        Failure discovered in Jupyter notebook on 2018-04-06
        """
        z_shape = (1, 1)
        z_1 = Input(shape=z_shape)
        f = Model(z_1, Lambda(lambda x: 2 * x)(z_1), name='f')

        # @formatter:off
        ground_truth_trajectories = [
            [ 2.0, 4.0, 8.0, 16.0, 32.0],
            [ 3.0, 4.0, 5.0,  6.0,  8.0, 10.0, 12.0],
        ]
        # @formatter:on

        delta_t_bounds = (1, 3)
        _sched_sampling_temperature = 0.0
        _exploration_p = 0.0

        expected_effective_z_hat_lengths = [4, 2]

        expected_jump_timesteps = [
            [1, 2, 3, 4],
            [3, 6, 6, 6]
        ]

        expected_mean_loss = 0.0
        self.run_test(z_shape, ground_truth_trajectories, f, delta_t_bounds,
                      _sched_sampling_temperature,
                      _exploration_p,
                      expected_mean_loss,
                      expected_effective_z_hat_lengths,
                      expected_jump_timesteps)
