import os

import numpy as np
from moviepy import editor as mpy

from asi.callbacks.train_callback import TrainCallback
from utils import math_util, mplvid


class PredictionVideoCallback(TrainCallback):
    def __init__(self, logger, output_directory, true_x_batch,
                 every_n_epochs, fps=10, scale_factor=10,
                 predictor_data_format='channels_last',
                 show_z_only=False, show_bar_charts=True,
                 n_videos=1,
                 video_codec=None
                 ):
        self.logger = logger
        self.output_directory = output_directory
        self.true_x_batch = true_x_batch
        self.every_n_epochs = every_n_epochs
        self.fps = fps
        self.scale_factor = scale_factor
        self.predictor_data_format = predictor_data_format
        self.show_z_only = show_z_only
        self.show_bar_charts = show_bar_charts
        self.n_videos = n_videos
        self.video_codec = video_codec

        if self.n_videos > len(self.true_x_batch):
            raise ValueError('n_videos cannot be greater than the number of videos in '
                             'true_x_batch')

    def on_batch_end(self, trainer, predictor, i_batch):
        # self.generate_video(i_batch * 10000, predictor, trainer)
        pass

    def on_epoch_end(self, trainer, predictor, i_epoch):
        if i_epoch % self.every_n_epochs == 0:
            self.make_videos(i_epoch, predictor, trainer)
        else:
            self.make_videos(i_epoch, predictor, trainer, n=1)

    def make_videos(self, i_epoch, predictor, trainer, n=None):
        if n is None:
            n = self.n_videos

        # Keep number of trajectories per video the same for all n
        trajectories_per_video = len(self.true_x_batch) // self.n_videos
        for i_video in range(n):
            x_batch = self.true_x_batch[
                      i_video * trajectories_per_video:
                      (i_video + 1) * trajectories_per_video
                      ]
            self.generate_video(x_batch, i_epoch, i_video, predictor, trainer)

    def generate_video(self, x_batch, i_epoch, i_video, predictor, trainer):
        self.logger.info('Generating analysis video....')
        video_frames = []
        for true_x_trajectory in x_batch:
            if self.predictor_data_format == 'channels_first':
                prediction_input = np.transpose(true_x_trajectory, (0, 3, 1, 2))
            elif self.predictor_data_format == 'channels_last':
                prediction_input = true_x_trajectory
            else:
                raise ValueError('Illegal data_format')

            trajectory_lengths = [prediction_input.shape[0]]
            metrics = predictor.analyze_batch(trainer.sess,
                                              x_padded=prediction_input[np.newaxis],
                                              trajectory_lengths=trajectory_lengths,
                                              fetch_z=True,
                                              fetch_jump_steps=True,
                                              fetch_effective_z_hat_lengths=True)
            z = metrics['z'][0]

            jump_timesteps = metrics['jump_timesteps'][0]

            n_predicted_frames = int(metrics['effective_z_hat_lengths'])

            z_hat = predictor.predict_n_steps(
                prediction_input[0:1],
                n=n_predicted_frames,
                sess=trainer.sess)

            if self.predictor_data_format == 'channels_first':
                z_hat = np.transpose(z_hat, (0, 1, 3, 4, 2))

            z_hat = z_hat[0]

            skip_values = np.diff(np.append([0], jump_timesteps))

            # Add misisng frames at the end
            skip_values[-1] += z.shape[0] - np.sum(skip_values)

            predicted_video = np.repeat(z_hat, repeats=skip_values, axis=0)

            true_skip_video = np.repeat(
                true_x_trajectory[jump_timesteps],
                skip_values,
                axis=0
            )

            if (true_x_trajectory.shape[1] % z.shape[1]) == 0:
                scale_factor = true_x_trajectory.shape[1] // z.shape[1]
                z = scale_up(z, scale_factor)
                predicted_video = scale_up(predicted_video, scale_factor)

            # Pad trajectories
            true_x_trajectory = np.pad(true_x_trajectory,
                                       ((0, 0), (2, 2), (2, 2), (0, 0)),
                                       mode='constant')
            true_skip_video = np.pad(true_skip_video,
                                     ((0, 0), (2, 2), (2, 2), (0, 0)),
                                     mode='constant')
            z = np.pad(z, ((0, 0), (2, 2), (2, 2), (0, 0)),
                       mode='constant')
            predicted_video = np.pad(predicted_video,
                                     ((0, 0), (2, 2), (2, 2), (0, 0)),
                                     mode='constant')
            if self.show_z_only:
                combined_frames = np.block(
                    [[[z], [predicted_video]]])
            else:
                combined_frames = np.block(
                    [[[true_x_trajectory], [true_skip_video]],
                     [[z], [predicted_video]]])

            combined_frames = scale_up(combined_frames, self.scale_factor)
            combined_frames = np.round(255 * combined_frames).astype(np.uint8)

            if self.show_bar_charts:
                # TODO: Replace with actual predictions
                y_hat_logits = np.zeros(shape=(n_predicted_frames, 1))

                y_hat_ps = math_util.softmax(y_hat_logits, axis=-1)

                viz_frame_height = combined_frames.shape[1]
                viz_frame_width = viz_frame_height

                xs = np.arange(y_hat_ps.shape[1])
                y_hat_video = mplvid.get_bar_chart_video(xs,
                                                         y_hat_ps,
                                                         viz_frame_height,
                                                         viz_frame_width,
                                                         dpi=120,
                                                         style='default',
                                                         ylim=(0.0, 1.0),
                                                         xticks=xs,
                                                         xticklabels=xs)

                y_hat_video = np.repeat(y_hat_video, repeats=skip_values, axis=0)

                combined_frames = np.concatenate((combined_frames, y_hat_video),
                                                 axis=-2)

            video_frames.extend(list(combined_frames))

        clip = mpy.ImageSequenceClip(video_frames, fps=self.fps)
        self.save_clip(clip, i_epoch, i_video)

    def save_clip(self, clip, i_epoch, i_video):

        if self.video_codec == 'png':
            video_extension = 'avi'
        elif self.video_codec is None:
            video_extension = 'mp4'
        else:
            raise ValueError('Unknown codec')
        filepath = os.path.join(self.output_directory,
                                '{}_{}.{}'.format(i_epoch, i_video, video_extension))
        self.logger.info('Writing video file {}'.format(filepath))
        clip.write_videofile(filepath, codec=self.video_codec,
                             fps=self.fps, progress_bar=False, verbose=False)


def scale_up(img, factor):
    return np.repeat(np.repeat(img, factor, axis=-3), factor, axis=-2)
