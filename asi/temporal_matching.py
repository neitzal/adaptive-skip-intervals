from collections import namedtuple

import tensorflow as tf

from utils import tf_util


def select_frame_slices(i_actual, all_frames, delta_t_bounds):
    """
    Returns consecutive slices from every row of all_frames.
    The starting positions for each row are given by i_actual + delta_t_bounds[0].

    If the slice is too long for the row, the last element of the respective row is
    repeated.

    :param i_actual: index array (tf.Tensor of rank 1) whose length is the batch size.
                     Indicates the previous step's positions within each trajectory.
    :param all_frames: Base array of shape (batch_size, trajectory_length)
    :param delta_t_bounds: 2-tuple (min_delta_t, max_delta_t) which specifies the minimum
                           time jump and the maximum time jump (both inclusive!)

    Example:

    select_frame_slices(
      i_actual=[1, 2, 0],
      all_frames=[[ 0,  1,  2,  3],
                  [10, 11, 12, 13],
                  [20, 21, 22, 23]]
      delta_t_bounds=(1, 2))

      ==

    [[ 2,  3],
     [13, 13],
     [21, 22]]
    """
    selected_frames = [tf_util.index_with_arrays(all_frames,
                                                 tf.range(tf.shape(i_actual)[0]),
                                                 tf.minimum(
                                                     i_actual + offset,
                                                     tf.shape(all_frames)[1] - 1)
                                                 )
                       for offset in range(delta_t_bounds[0],
                                           delta_t_bounds[1] + 1)]
    selected_frames = [tf.expand_dims(frame, axis=1) for frame in selected_frames]
    selected_frames = tf.concat(selected_frames, axis=1, name='selected_frames')
    return selected_frames


def reduce_losses(loss_scalars, reduction_ndim):
    """
    From the collection of loss scalars, get the mean over the last reduction_ndim axes
    """
    reduction_axes = tuple(range(-1, -reduction_ndim - 1, -1))
    losses = tf.reduce_mean(loss_scalars,
                            axis=reduction_axes)
    return losses


def mix_with_random_values(original_indices, max_index, exploration_p, batch_size,
                           seed=None):
    """
    Returns a noisy variant of original_indices where randomly selected rows
    (each selected independently with probability exploration_p) are replaced
    by discrete uniform values between 0 and max_index (exclusive!)
    """
    random_determiners = tf.random_uniform((batch_size,),
                                           0, 1.0,
                                           dtype=tf.float32,
                                           seed=seed,
                                           name='random_determiners',
                                           )
    random_selection = tf.random_uniform((batch_size,), 0,
                                         max_index,
                                         dtype=tf.int32,
                                         # Change seed to avoid picking the same values
                                         seed=seed + 345 if seed is not None else None,
                                         name='random_skip')
    selection_indices = tf.where(random_determiners < exploration_p,
                                 random_selection,
                                 original_indices)

    return selection_indices


def cut_values(values, keep_array):
    """

    keep_array is a boolean array matching the length of values, which
        specifies the values to count
    """
    effective_values = tf.to_float(keep_array) * values
    kept_count = tf.reduce_sum(tf.cast(keep_array, tf.int32))
    value_sum = tf.reduce_sum(effective_values)
    return kept_count, value_sum


def accumulate_in_loop(acc, index, value):
    """
    Get an updated version of the accumulator, which
    is equal to the old accumulator, but is incremented by "value" at acc[index]

    :param acc: accumulator tensor, of dimension at least 1D
    :param index: scalar int tensor, index at which to update.
    :param value: value which is added to acc at index. Must have shape == acc.shape[1:]
    :return: updated acc
    """
    indices = index[tf.newaxis, tf.newaxis]
    shaped_diff = tf.scatter_nd(indices=indices,
                                updates=tf.expand_dims(value, axis=0),
                                shape=tf.shape(acc))
    return tf.add(acc, shaped_diff)


def predict_match_interleaved(z,
                              dynamics_f,
                              delta_t_bounds,
                              loss_fn,
                              trajectory_lengths,
                              exploration_p,
                              scheduled_sampling_temperature,
                              random_seed=None,
                              parallel_iterations=10,
                              ):
    """
    :return: tuple of tf-Tensors (mean_z_loss, effective_z_hat_lengths, z_hat)
             mean_z_loss: a scalar which represents the mean z-loss for the
                          given inputs. This is computed on the fly because
                          it is needed for temporal matching
             effective_z_hat_length_counter: a tensor of shape (batch_size,)
                                             which contains the actual lengths
                                             of the predicted trajectories until
                                             the predicted frame matches the last
                                             actual frame.
             z_hat: A tensor of shape (batch_size, max_trajectory_length*,) + z_shape
                    which contains predicted latent states. Since we use dynamic
                    temporal matching, there is not necessarily a 1-to-1 correspondence
                    between time steps of z_hat and z.
                    *max_trajectory_length is the number of steps in the longest
                    predicted trajectory. Shorter trajectories are padded with zeros at
                    the end.

    """
    dynamic_z_shape = tf.shape(z)
    batch_size = dynamic_z_shape[0]
    max_trajectory_length = tf.reduce_max(trajectory_lengths)
    i_pred = tf.constant(0, tf.int32, shape=(), name='i_pred')
    i_actual = tf.zeros(shape=dynamic_z_shape[:1], dtype=tf.int32,
                        name='i_actual')
    effective_z_hat_length_counter = tf.zeros_like(i_actual)

    loss_acc = tf.constant(0.0, tf.float32, shape=(), name='loss_acc')
    loss_count = tf.constant(0, tf.int32, shape=(), name='loss_count')

    transpose_order = [1, 0] + list(range(2, len(z.shape)))
    z_transposed = tf.transpose(z, transpose_order)

    jump_timestep_acc = tf.zeros(tf.shape(z_transposed)[:2], dtype=tf.int32,
                                 name='jump_timestep_acc')

    z_hat_transposed_acc = tf.zeros(
        # Transposed - therefore the trajectory steps are on axis 0, batch is on axis 1
        tf.shape(z_transposed),
        dtype=tf.float32,
        name='z_hat_transposed_acc'
    )

    z_step_losses_acc = tf.zeros(
        # Transposed - therefore the trajectory steps are on axis 0, batch is on axis 1
        (max_trajectory_length - 1,
         batch_size,
         delta_t_bounds[1] - delta_t_bounds[0] + 1),
        dtype=tf.float32,
        name='z_step_losses_acc'
    )

    LoopState = namedtuple('LoopState',
                           'i_pred i_actual loss_acc loss_count '
                           'effective_z_hat_length_counter '
                           'z_hat_transposed_acc jump_timestep_acc '
                           'z_step_losses_acc')

    def loop_body(v):
        selected_frames = select_frame_slices(v.i_actual, z, delta_t_bounds)

        new_seed = random_seed + 345 if random_seed is not None else None
        scheduled_sampling_randoms = tf.random_uniform((batch_size,), 0.0, 1.0,
                                                       # avoid same random seed as for
                                                       # temporal matching exploration,
                                                       # because they are both called once
                                                       # per iteration
                                                       seed=new_seed,
                                                       name='scheduled_sampling_random')

        current_z_transpd = tf_util.index_with_arrays(z_transposed,
                                                      tf.minimum(v.i_actual,
                                                                 trajectory_lengths - 1),
                                                      tf.range(batch_size))

        f_input = tf.cond(
            tf.equal(v.i_pred, 0),
            true_fn=lambda: current_z_transpd,
            false_fn=lambda: tf.where(
                scheduled_sampling_randoms < scheduled_sampling_temperature,
                current_z_transpd,
                v.z_hat_transposed_acc[v.i_pred - 1]
            ))
        # f_input = tf.expand_dims(f_input, axis=1, name='f_input')
        prediction = dynamics_f(f_input)

        # # Update z_hat
        updated_z_hat_transposed_acc = accumulate_in_loop(v.z_hat_transposed_acc,
                                                          v.i_pred, prediction)


        # Losses
        loss_scalars = loss_fn(selected_frames,
                               tf.expand_dims(prediction, axis=1))

        losses = reduce_losses(loss_scalars, len(z.shape) - 2)
        losses_argmin = tf.argmin(losses, output_type=tf.int32, axis=-1)

        # Inject exploration
        selection_indices = mix_with_random_values(losses_argmin,
                                                   (delta_t_bounds[1] - delta_t_bounds[0]
                                                    + 1),
                                                   exploration_p, batch_size,
                                                   seed=random_seed)

        skip_values = selection_indices + delta_t_bounds[0]

        # Constrain skip values such that you can never jump outside of trajectory,
        # But you jump at least the minimum length
        skip_values = tf.minimum(skip_values, trajectory_lengths - v.i_actual - 1)
        skip_values = tf.maximum(skip_values, delta_t_bounds[0])

        selected_losses = tf_util.index_with_arrays(losses,
                                                    tf.range(batch_size),
                                                    selection_indices)

        i_actual_new = v.i_actual + skip_values
        within_bounds = i_actual_new < trajectory_lengths
        effective_loss_count, loss = cut_values(selected_losses, within_bounds)

        # Count the z_hat lengths where the predictions are still within the
        # actual trajectory bounds
        new_effective_z_hat_length_counter = tf.where(
            within_bounds,
            v.effective_z_hat_length_counter + 1,
            v.effective_z_hat_length_counter
        )
        filtered_skip_values = tf.where(within_bounds,
                                        skip_values,
                                        tf.zeros_like(skip_values))

        new_jump_timestep_acc = accumulate_in_loop(v.jump_timestep_acc,
                                                   v.i_pred, filtered_skip_values)

        new_z_step_losses_acc = accumulate_in_loop(v.z_step_losses_acc,
                                                   v.i_pred, losses)

        next_loop_state = LoopState(v.i_pred + 1,
                                    i_actual_new,
                                    v.loss_acc + loss,
                                    v.loss_count + effective_loss_count,
                                    new_effective_z_hat_length_counter,
                                    updated_z_hat_transposed_acc,
                                    new_jump_timestep_acc,
                                    new_z_step_losses_acc
                                    )
        return [next_loop_state]

    def cond(v):
        reached_eot = tf.greater_equal(v.i_actual + 1, trajectory_lengths,
                                       name='reached_eot')
        return tf.logical_not(tf.reduce_all(reached_eot))

    loop_var = LoopState(i_pred,
                         i_actual,
                         loss_acc,
                         loss_count,
                         effective_z_hat_length_counter,
                         z_hat_transposed_acc,
                         jump_timestep_acc,
                         z_step_losses_acc)

    final_state, = tf.while_loop(cond,
                                 loop_body,
                                 loop_vars=[loop_var],
                                 parallel_iterations=parallel_iterations)

    mean_z_loss = final_state.loss_acc / tf.to_float(final_state.loss_count)

    # Truncate unused space at the end of z_hat's trajectories
    max_effective_z_hat_length = tf.reduce_max(final_state.effective_z_hat_length_counter)
    z_hat_transposed = final_state.z_hat_transposed_acc[:max_effective_z_hat_length]

    z_hat = tf.transpose(z_hat_transposed, perm=transpose_order,
                         name='z_hat')

    # Truncate unused jump_timesteps as well
    jump_timesteps_transposed = final_state.jump_timestep_acc[
                                :max_effective_z_hat_length]
    jump_timesteps = tf.transpose(jump_timesteps_transposed, [1, 0])
    jump_timesteps = tf.cumsum(jump_timesteps, axis=1, name='jump_timesteps')

    z_step_losses = tf.transpose(final_state.z_step_losses_acc, [1, 0, 2])

    return (mean_z_loss,
            final_state.effective_z_hat_length_counter,
            z_hat,
            jump_timesteps,
            z_step_losses)
