"""
Task-specific accuracy scores for the predictions
"""
import numpy as np
from scipy import signal


def fubo_ball_position(frame):
    frame = _prepare_fubo_frame(frame)
    ball_map = _get_fubo_ball_map(frame)
    if np.max(ball_map) <= 0.04:
        return None
    return np.array(np.unravel_index(np.argmax(ball_map),
                                     frame.shape[:2]))


def fubo_label(frame):
    assert frame.shape == (126, 75, 3), 'frame.shape == {}'.format(
        frame.shape)  # Function assumes this shape
    position = fubo_ball_position(frame)
    if position is None:
        return None

    pos_x = position[1]
    if 0 <= pos_x <= 74:
        return int(pos_x / 15)
    else:
        return None

def _get_fubo_ball_map(frame):
    """Specific method to extract a map of where the ball is"""

    assert np.max(frame) <= 1
    assert np.min(frame) >= 0

    # Blue-filter: computes (1 - R)*(1 - G)*B
    blue_filtered = (1 - frame[:, :, 0]) * (1 - frame[:, :, 1]) * frame[:, :, 2]

    # Threshold to keep only the ball
    blue_filtered = np.where(blue_filtered < 0.3,
                             0,
                             blue_filtered)

    return blue_filtered


def _prepare_fubo_frame(frame):
    assert frame.shape == (126, 75, 3), 'frame.shape == {} '.format(frame.shape)
    if frame.dtype == np.uint8:
        frame = frame / 255
    frame = frame[:-2]
    return frame


def predict_fubo_label(trajectory_frames):
    ball_maps = np.array([_get_fubo_ball_map(_prepare_fubo_frame(frame))
                          for frame in trajectory_frames])
    lower_part = ball_maps[:, -10:, :]
    bucket_sums = [
        lower_part[:, :, 15 * k:15 * (k + 1)].sum()
        for k in range(5)
    ]
    epsilon = 1e-3
    n_labels = 5
    return (np.array(bucket_sums) + epsilon / n_labels) / (
            np.sum(lower_part) + epsilon)


def fubo_fair_accuracy(x_trajectory_batch, x_hat_trajectory_batch, trajectory_lengths):
    """
    This accuracy function is realistic in the sense that we don't take a maximum
    over scores or anything. We are determining the label only based on the prediction
    and then *afterwards* check whether it fits the ground truth!
    """
    results = []
    for x_trajectory, x_hat_trajectory, trajectory_length in zip(
            x_trajectory_batch,
            x_hat_trajectory_batch,
            trajectory_lengths):
        prediction = predict_fubo_label(x_hat_trajectory)

        true_label = fubo_label(x_trajectory[trajectory_length - 1])

        if true_label is not None:
            score = prediction[true_label]
            results.append(score)

    return np.mean(results)


def fubo_ball_maintenance_frames(x_trajectory_batch, x_hat_trajectory_batch,
                                 trajectory_lengths):
    """Measures how many frames the ball is kept alive"""
    ball_lost_indices = []
    for x_trajectory in x_hat_trajectory_batch:
        try:
            ball_lost_index = next(i for i, frame in enumerate(x_trajectory)
                                   if fubo_ball_position(frame) is None)
        except StopIteration:
            ball_lost_index = len(x_trajectory)
        ball_lost_indices.append(ball_lost_index)
    return np.mean(ball_lost_indices)


# @formatter:off
rr_room_smoothing_kernel = np.array([[1/16, 1/8, 1/16],
                                     [1/8,  1/4, 1/8],
                                     [1/16, 1/8, 1/16]])

rr_room_overlowing_kernel = np.ones((3, 3), np.uint8)
# @formatter:on


def predict_rr_label_probas(first_gt_frame, trajectory):
    if first_gt_frame.dtype == np.uint8:
        first_gt_frame = first_gt_frame.copy() / 255

    if trajectory.dtype == np.uint8:
        trajectory = trajectory.copy() / 255

    threshold_red = 0.15
    red_room = (first_gt_frame[:, :, 0]
                * (1 - first_gt_frame[:, :, 1])
                * (1 - first_gt_frame[:, :, 2])) > threshold_red

    # overflow into white area
    red_room = get_preceding_room_area(first_gt_frame, red_room)

    # Smooth everything
    red_room = signal.correlate2d(red_room, rr_room_smoothing_kernel,
                                  boundary='symm', mode='same')

    threshold_blue = 0.15
    blue_room = ((1 - first_gt_frame[:, :, 0])
                 * (1 - first_gt_frame[:, :, 1])
                 * (first_gt_frame[:, :, 2])) > threshold_blue

    # overflow into white area
    blue_room = get_preceding_room_area(first_gt_frame, blue_room)

    blue_room = signal.correlate2d(blue_room, rr_room_smoothing_kernel,
                                   boundary='symm', mode='same')

    threshold_runner = 0.25
    runner = ((1 - trajectory[:, :, :, 0])
              * (trajectory[:, :, :, 1])
              * (1 - trajectory[:, :, :, 2])) > threshold_runner

    weight_red = np.sum(red_room[np.newaxis] * runner)
    weight_blue = np.sum(blue_room[np.newaxis] * runner)

    epsilon = 1e-4
    p_red = (weight_red + epsilon) / (weight_blue + weight_red + 2 * epsilon)
    return np.array([p_red, 1 - p_red])


def get_preceding_room_area(first_gt_frame, target_room):
    """
    For RR task, given a target room map and the first frame, get some pixels of the last
    room adjacent to the target room
    """
    overflow_area = target_room
    for i in range(8):
        overflow_area = signal.correlate2d(overflow_area, rr_room_overlowing_kernel,
                                           boundary='symm', mode='same')
        overflow_area = np.clip(overflow_area, 0, 1)
        overflow_area = np.logical_and(overflow_area > 0.1,
                                       first_gt_frame.mean(axis=-1) > 0.9)
    target_room = np.logical_or(target_room, overflow_area)
    return target_room


def rr_fair_accuracy(x_trajectory_batch, x_hat_trajectory_batch, trajectory_lengths):
    results = []

    for x_trajectory, x_hat_trajectory, trajectory_length in zip(
            x_trajectory_batch,
            x_hat_trajectory_batch,
            trajectory_lengths):
        # This one can be used for predictions as it is observable
        first_gt_frame = x_trajectory[0]

        true_label_probas = predict_rr_label_probas(first_gt_frame, x_trajectory)
        true_label = np.argmax(true_label_probas)
        prediction = predict_rr_label_probas(first_gt_frame, x_hat_trajectory)
        score = prediction[true_label]
        results.append(score)

    return np.mean(results)


def has_runner(frame):
    if frame.dtype == np.uint8:
        frame = frame / 255.

    # runner_color == (0.0, 0.8, 0.0)
    runner_map = ((1 - frame[:, :, 0])
                  * (1 - 1.5 * (frame[:, :, 1] - 0.8) ** 2)
                  * (1 - frame[:, :, 2]))
    return np.any(runner_map > 0.25)


def rr_runner_maintenance_frames(x_trajectory_batch, x_hat_trajectory_batch,
                                 trajectory_lengths):
    """Measures how many frames the runner is kept alive"""
    runner_lost_indices = []
    for x_trajectory in x_hat_trajectory_batch:
        try:
            runner_lost_index = next(i for i, frame in enumerate(x_trajectory)
                                     if not has_runner(frame))
        except StopIteration:
            runner_lost_index = len(x_trajectory)
        runner_lost_indices.append(runner_lost_index)
    return np.mean(runner_lost_indices)
