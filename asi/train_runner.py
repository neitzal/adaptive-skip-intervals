import json
import logging
import time
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from asi.callbacks.train_callback import TrainCallback
from asi.models.asi_model import ASIModel
from utils.misc import flatten_list_one_level
from utils.tf_util import make_count_variable, make_count_resetter, \
    make_count_incrementer


class Trainer:
    def __init__(self, sess=None, logger=None):
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.saver = None  # Needs to be initialized after the model has been built
        self.model_path = None

        if logger is None:
            logger = logging.getLogger(__file__)
            logger.setLevel(logging.INFO)

        self.logger = logger

    def save_model(self):
        if self.saver is None:
            raise ValueError('Cannot save model, saver is not initialized.')
        self.saver.save(self.sess, self.model_path)
        self.logger.info('Model saved at prefix {}'.format(self.model_path))

    def run(self,
            train_iterator_init_op,
            valid_iterator_init_op,
            predictor: ASIModel,
            n_epochs_max,
            task_specific_metrics,
            progress_filepath,
            step_data_filepath,
            callbacks: List[TrainCallback] = None,
            saver_max_to_keep=5,
            model_path=None):
        """
        :param train_iterator_init_op: tf-operation which initializes the train-iterator
        :param valid_iterator_init_op: tf-operation which initializes the valid-iterator
        :param predictor: Instance of ASIModel
        :param progress_filepath: Path to progress file
        :param callbacks: TrainCallback objects whose methods will be invoked at the
                          appropriate times
        :param saver_max_to_keep: max_to_keep argument for tf.Saver
        :param model_path: Path from where to restore model and where to save models
        :return:
        """

        self.model_path = model_path

        if callbacks is None:
            callbacks = []

        global_step = make_count_variable('global_step')
        i_epoch = make_count_variable('i_epoch')
        i_batch = make_count_variable('i_batch')

        increment_global_step = make_count_incrementer(global_step)
        increment_epoch = make_count_incrementer(i_epoch)
        increment_batch = make_count_incrementer(i_batch)

        reset_batch_counter = make_count_resetter(i_batch)

        init_var_op = tf.global_variables_initializer()

        self.saver = tf.train.Saver(max_to_keep=saver_max_to_keep)
        try:
            self.saver.restore(self.sess, model_path)
            self.logger.info('Restored model from path {}'.format(model_path))
            self.logger.info('Resuming asi from '
                             'i_epoch={}, '
                             'i_batch={}'.format(
                self.sess.run(i_epoch),
                self.sess.run(i_batch),
            ))
        except tf.errors.NotFoundError:
            self.logger.info('No model to restore at path {}'.format(model_path))
            self.sess.run(init_var_op)
            self.logger.info('Initialized fresh model.')

        with self.sess.as_default():
            self.sess.graph.finalize()

            try:

                while i_epoch.eval() < n_epochs_max:

                    self.logger.info('===============================================')
                    self.logger.info('Train i_epoch={}'.format(i_epoch.eval()))
                    self.logger.info('===============================================')
                    training_durations = []
                    all_train_metrics = []

                    self.sess.run(train_iterator_init_op)

                    # Skip values if i_batch > 0 (i.e. we are resuming from checkpoint)
                    n_skip = i_batch.eval()
                    for i_skip in range(i_batch.eval()):
                        self.sess.run([predictor.trajectory_lengths])
                    if n_skip > 0:
                        self.logger.info('Done skipping to batch {}. '
                                         'Resuming asi now.'.format(n_skip))

                    while True:
                        try:
                            tic = time.time()
                            predictor.train_on_trajectories(self.sess)
                            toc = time.time()

                            # self.logger.info('Batch duration: {:.4f}'.format(toc - tic))

                            all_train_metrics.append(predictor.train_analytics)
                            training_durations.append(toc - tic)
                            self.sess.run(increment_global_step)
                            self.sess.run(increment_batch)

                            for callback in callbacks:
                                callback.on_batch_end(trainer=self,
                                                      predictor=predictor,
                                                      i_batch=i_batch.eval())
                        except tf.errors.OutOfRangeError:  # Epoch over
                            self.logger.info('Epoch finished. '
                                             'i_batch={}'.format(i_batch.eval()))
                            break

                    self.sess.run(reset_batch_counter)

                    i_finished_epoch = i_epoch.eval()
                    for callback in callbacks:
                        callback.on_epoch_end(trainer=self,
                                              predictor=predictor,
                                              i_epoch=i_finished_epoch,
                                              )

                    self.sess.run(increment_epoch)

                    self.save_model()

                    self.logger.info('Evaluating accuracy on validation data ...')
                    batch_sizes = []
                    all_valid_metrics = []
                    all_valid_jump_timesteps = []
                    all_valid_z_step_losses = []

                    all_effective_z_hat_lengths = [m['effective_z_hat_lengths']
                                                   for m in all_train_metrics]
                    new_max_depth = int(round(
                        np.percentile(all_effective_z_hat_lengths, 99.))) + 1

                    self.logger.info('new_max_depth: {}'.format(new_max_depth))
                    all_valid_metrics.append(
                        {'prediction_max_depth': new_max_depth})
                    predictor.prediction_max_depth = new_max_depth

                    self.sess.run(valid_iterator_init_op)

                    while True:
                        try:
                            validation_step(predictor, batch_sizes, all_valid_metrics,
                                            all_valid_jump_timesteps,
                                            all_valid_z_step_losses,
                                            task_specific_metrics,
                                            self.sess)

                            if len(all_valid_metrics) % 100 == 0:
                                self.logger.info('validation_step {} completed'.format(
                                    len(all_valid_metrics)))

                        except tf.errors.OutOfRangeError:
                            # Exception indicates that loop over tfrecord is completed
                            break

                    self.logger.info('Done evaluating accuracy on validation data.')

                    all_train_metrics.append(
                        {'batch_training_time':
                             float(np.mean(training_durations))
                         } )

                    if i_epoch.eval() % 4 == 0:
                        write_step_data(all_valid_jump_timesteps,
                                        all_valid_z_step_losses,
                                        (step_data_filepath
                                         + '_ep_{}.txt'.format(i_epoch.eval())))

                    write_metrics(global_step.eval(),
                                  i_finished_epoch,
                                  all_train_metrics,
                                  all_valid_metrics,
                                  training_durations,
                                  progress_filepath)

                    for callback in callbacks:
                        callback.on_validation_end(trainer=self,
                                                   predictor=predictor,
                                                   train_metrics=all_train_metrics,
                                                   valid_metrics=all_valid_metrics,
                                                   i_epoch=i_finished_epoch)

            except KeyboardInterrupt:
                self.logger.info('Training interrupted')
                self.save_model()
                for callback in callbacks:
                    callback.on_keyboard_interrupt(self)

        for callback in callbacks:
            callback.on_training_end(self)

        return self.sess


def write_metrics(global_step, i_epoch, train_metrics, valid_metrics,
                  training_durations, progress_filepath):
    train_metrics = pd.DataFrame(train_metrics).add_suffix('_train')
    valid_metrics = pd.DataFrame(valid_metrics).add_suffix('_valid')

    train_stats = {
        'global_step': int(global_step),
        'i_epoch': int(i_epoch),
        'batch_training_time': float(np.mean(training_durations)),
        **(train_metrics.mean(axis=0, skipna=True).to_dict()),
        **(valid_metrics.mean(axis=0, skipna=True).to_dict()),
    }

    try:
        train_stats_str = json.dumps(train_stats)
    except TypeError:
        raise TypeError('Not JSON-serializable: ' + str(train_stats))

    with open(progress_filepath, 'a') as f:
        f.write('{}\n'.format(train_stats_str))


def validation_step(predictor: ASIModel, batch_sizes, all_valid_metrics,
                    all_valid_jump_timesteps, all_valid_z_step_losses,
                    task_specific_metrics,
                    sess):
    (frames_valid,
     trajectory_lengths) = sess.run([predictor.x,
                                     predictor.trajectory_lengths])

    batch_sizes.append(len(frames_valid))

    valid_metrics = predictor.analyze_batch(sess,
                                            x_padded=frames_valid,
                                            trajectory_lengths=trajectory_lengths,
                                            fetch_jump_steps=True,
                                            fetch_z_step_losses=True,
                                            fetch_z_hat=True,
                                            )

    # Clean up unwanted output from metrics
    all_valid_jump_timesteps.append(valid_metrics['jump_timesteps'])
    all_valid_z_step_losses.append(valid_metrics['z_step_losses'])

    # ### Get predicted batch ###
    predicted_batch = []
    for trajectory in frames_valid:
        predicted_frames = predictor.predict_n_steps(trajectory[0][np.newaxis],
                                                     predictor.prediction_max_depth,
                                                     sess)[0]
        predicted_batch.append(predicted_frames)
    predicted_batch = np.asarray(predicted_batch)
    # ###########################

    for key, metric_fn in task_specific_metrics.items():
        valid_metrics[key + '_matched'] = metric_fn(frames_valid,
                                                    valid_metrics['z_hat'],
                                                    trajectory_lengths)

        valid_metrics[key] = metric_fn(frames_valid,
                                       predicted_batch,
                                       trajectory_lengths)

    del valid_metrics['z_hat']
    del valid_metrics['jump_timesteps']
    del valid_metrics['z_step_losses']

    all_valid_metrics.append(valid_metrics)


def write_step_data(all_valid_jump_timesteps,
                    all_valid_z_step_losses,
                    filepath):
    # Flatten out batches
    all_valid_jump_timesteps = flatten_list_one_level(all_valid_jump_timesteps)
    all_valid_z_step_losses = flatten_list_one_level(all_valid_z_step_losses)

    with open(filepath, 'a') as f:
        for jump_timesteps, all_valid_z_step_losses in zip(all_valid_jump_timesteps,
                                                           all_valid_z_step_losses):
            f.write(','.join('{}'.format(s) for s in jump_timesteps) + '\n')

            min_losses = np.min(all_valid_z_step_losses, axis=-1)
            f.write(','.join('{}'.format(zl) for zl in min_losses) + '\n')
