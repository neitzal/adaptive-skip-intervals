"""
Code to generate timeline-plots during training
"""

import matplotlib

from utils import math_util

matplotlib.use('Agg')

from matplotlib.path import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os

import numpy as np

from asi.callbacks.train_callback import TrainCallback


class TimelineCallback(TrainCallback):
    def __init__(self, logger, output_directory, true_x_batch,
                 every_n_epochs, delta_t_bounds,
                 variations,
                 predictor_data_format='channels_last',
                 paper_mode=False):
        self.logger = logger
        self.output_directory = output_directory
        self.true_x_batch = true_x_batch
        self.every_n_epochs = every_n_epochs
        self.predictor_data_format = predictor_data_format
        self.delta_t_bounds = delta_t_bounds
        self.variations = variations
        self.paper_mode = paper_mode

    def on_batch_end(self, trainer, predictor, i_batch):
        pass

    def on_epoch_end(self, trainer, predictor, i_epoch):
        if i_epoch % self.every_n_epochs == 0:
            self.generate_timeline(i_epoch, predictor, trainer)
        else:
            self.generate_timeline(i_epoch, predictor, trainer, n=1)

    def generate_timeline(self, i_epoch, predictor, trainer, n=None):
        if n is not None:
            batch = self.true_x_batch[:n]
        else:
            batch = self.true_x_batch

        self.logger.info('Generating timeline...')
        for i_trajectory, true_x_trajectory in enumerate(batch):
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
                                              fetch_effective_z_hat_lengths=True,
                                              fetch_z_step_losses=True)
            z = metrics['z'][0]
            jump_timesteps = list(metrics['jump_timesteps'][0])
            z_step_losses = metrics['z_step_losses'][0]

            n_predicted_frames = int(metrics['effective_z_hat_lengths'])

            z_hat = predictor.predict_n_steps(
                prediction_input[0:1],
                n=n_predicted_frames,
                sess=trainer.sess)

            if self.predictor_data_format == 'channels_first':
                z_hat = np.transpose(z_hat, (0, 1, 3, 4, 2))

            z_hat = z_hat[0]

            for variation in self.variations:
                if self.paper_mode:
                    frames_per_part = 12

                    def partition(arr):
                        return [arr[i * frames_per_part:(i + 1) * frames_per_part]
                                for i in range(frames_per_part)]

                    z_parts = partition(z)

                    z_hat_parts = []
                    jump_timesteps_parts = []
                    z_step_losses_parts = []

                else:
                    z_parts = [z]
                    z_hat_parts = [z_hat]
                    jump_timesteps_parts = [jump_timesteps]
                    z_step_losses_parts = [z_step_losses]

                for (i_part,
                     (z,
                      z_hat,
                      jump_timesteps,
                      z_step_losses)) in enumerate(zip(z_parts,
                                                       z_hat_parts,
                                                       jump_timesteps_parts,
                                                       z_step_losses_parts)):
                    fig = self.get_timeline_figure(z, z_hat, jump_timesteps,
                                                   z_step_losses,
                                                   variation)
                    filename = '{}_{}_timeline_{}_{}.png'.format(i_epoch, i_trajectory,
                                                                 variation, i_part)
                    figure_path = os.path.join(self.output_directory, filename)
                    fig.savefig(
                        figure_path)
                    plt.close(fig)
                    self.logger.info('Timeline saved at {}'.format(figure_path))

    def get_timeline_figure(self, z, z_hat, jump_timesteps, z_step_losses,
                            variation, dpi=100):
        if variation not in ['A', 'B', 'C']:
            raise ValueError('Invalid variation: {}'.format(variation))

        losses_color = '#104080'
        if self.paper_mode:
            frame_width = 2.0
            frame_height = frame_width * (z.shape[-3] / z.shape[-2])
            horizontal_spacing = 0.05
            vertical_spacing = 0.05
            outer_margin = 0.5
        else:
            frame_width = 2.0
            frame_height = frame_width * (z.shape[-3] / z.shape[-2])
            horizontal_spacing = 0.4
            vertical_spacing = 2.0
            outer_margin = 1.0
        figure_width = (2 * outer_margin + z.shape[0] * frame_width
                        + (z.shape[0] - 1) * horizontal_spacing)
        figure_height = (2 * outer_margin + 2 * frame_height + vertical_spacing)
        font_size_inches = 0.1 * frame_height

        fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

        def translate_i_to_x(i):
            return (outer_margin + i * (frame_width + horizontal_spacing)) / figure_width

        def get_rect(i_x, i_y):
            x = translate_i_to_x(i_x)
            y = (outer_margin + i_y * (
                    frame_height + vertical_spacing)) / figure_height  # from bottom to top
            w = frame_width / figure_width
            h = frame_height / figure_height
            return [x, y, w, h]

        axs = []

        # --- Add ground truth images ---
        for i_z, z_frame in enumerate(z):
            _rect = get_rect(i_z, 1)
            ax = fig.add_axes(_rect)
            ax.imshow(z_frame.astype(float), vmin=0.0, vmax=1.0)
            axs.append(ax)

        # --- Add predictions and connections ---
        connections_ax = fig.add_axes(
            [0., (outer_margin + frame_height) / figure_height,
             1.0, vertical_spacing / figure_height])
        connections_ax.set_axis_off()

        arrows_ax = fig.add_axes([0.,
                                  outer_margin / figure_height,
                                  1.0,
                                  (frame_height + vertical_spacing) / figure_height])

        arrows_ax.set_axis_off()

        def get_strengths(losses):
            unscaled = math_util.softmax(-500.0 * losses)
            return unscaled

        def add_curve(i_from, i_to, linewidth, alpha):
            # start_x = ((outer_margin + 0.5 * frame_width
            #             + i_from * (frame_width + horizontal_spacing)) /
            #            figure_width)
            start_x = translate_i_to_x(i_from) + 0.5 * frame_width / figure_width
            start_y = 0
            end_x = translate_i_to_x(i_to) + 0.5 * frame_width / figure_width
            end_y = 1

            mid_y = (start_y + end_y) / 2
            pp1 = mpatches.PathPatch(
                Path([(start_x, start_y),
                      (start_x, mid_y),
                      (end_x, mid_y),
                      (end_x, end_y)],
                     [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                fc="none",
                # transform=connections_ax.transData,
                lw=linewidth,
                edgecolor=losses_color,
                alpha=alpha
            )
            connections_ax.add_patch(pp1)

        i_displayed = 0
        if variation == 'B':
            placement_timesteps = [0] + jump_timesteps[:-1]
        else:
            placement_timesteps = jump_timesteps

        for i_step, i_from in zip(placement_timesteps, [0] + jump_timesteps[:-1]):
            _rect = get_rect(i_step, 0)
            ax = fig.add_axes(_rect)
            img = z_hat[i_displayed]
            ax.imshow(img.astype(float), vmin=0.0, vmax=1.0)
            axs.append(ax)

            strengths = get_strengths(z_step_losses[i_displayed])

            for (connection_i, strength) in zip(range(
                    i_from + self.delta_t_bounds[0],
                    i_from + self.delta_t_bounds[1] + 1),
                    strengths):
                if connection_i >= z.shape[0]:
                    continue
                if variation == 'C':
                    add_curve(i_step, connection_i, linewidth=4.0, alpha=strength)
                else:
                    add_curve(i_step, connection_i, linewidth=20.0 * strength, alpha=0.5)
            i_displayed += 1

        # --- Add f arrows ---
        arrows_y = 0.5 * frame_height / (frame_height + vertical_spacing)
        arrow_head_width = (0.05 * frame_height) / figure_height
        arrow_head_length = (0.3 * horizontal_spacing) / figure_width
        arrow_lw = arrow_head_width / 8

        def add_arrow(i_from, i_to, description=None):
            x_from = translate_i_to_x(i_from + 1) - horizontal_spacing / figure_width
            dx = ((horizontal_spacing
                   + (i_to - i_from - 1) * (horizontal_spacing + frame_width))
                  / figure_width)

            y_from = arrows_y
            dy = 0

            arrows_ax.arrow(x_from, y_from, dx, dy,
                            width=arrow_lw, head_width=arrow_head_width,
                            head_length=arrow_head_length,
                            length_includes_head=True,
                            fc='#000000',  # Black arrow
                            ec='#000000'
                            )
            if description:
                arrows_ax.text(x_from + 0.5 * dx,
                               y_from + (0.2 * vertical_spacing) / figure_height,
                               description,
                               fontsize=20,
                               horizontalalignment='center',
                               verticalalignment='center'
                               )

        for i_from, i_to in zip(jump_timesteps[:-1], jump_timesteps[1:]):
            add_arrow(i_from, i_to,
                      # description='$f$'
                      )

        # Draw first arrow
        if variation != 'B':
            x_from = translate_i_to_x(0) + 0.5 * frame_width / figure_width
            x_to = translate_i_to_x(jump_timesteps[0])
            y_from = 1.0
            y_to = arrows_y

            path_patch = mpatches.PathPatch(
                Path([(x_from, y_from),
                      (x_from, y_to),
                      (x_to, y_to)],
                     [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                fc="none",
                # transform=connections_ax.transData,
                lw=arrow_lw * figure_height * dpi,
                edgecolor='#000000'
            )
            arrows_ax.add_patch(path_patch)

            # Arrow head for first arrow
            arrows_ax.arrow(x_to - 1e-6, y_to, 1e-6, 0,
                            width=arrow_lw, head_width=arrow_head_width,
                            head_length=arrow_head_length,
                            length_includes_head=True,
                            fc='#000000',  # Black arrow
                            ec='#000000'
                            )

            # Annotate first f
            arrows_ax.text(x_from,
                           (y_from + y_to) / 2,
                           'f',
                           fontsize=font_size_inches * dpi,
                           horizontalalignment='right',
                           verticalalignment='center',
                           family='serif'
                           )

        # Write annotation for losses
        if not self.paper_mode:
            connections_ax.text(translate_i_to_x(1),
                                0.5,
                                'Temporal matching',
                                fontsize=0.75 * font_size_inches * dpi,
                                horizontalalignment='left',
                                verticalalignment='center',
                                color=losses_color,
                                backgroundcolor='#ffffff',
                                family='serif'
                                )

        axs.append(connections_ax)
        axs.append(arrows_ax)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        return fig
