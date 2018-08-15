import os

import tensorflow as tf

from asi.train_runner import Trainer
from utils.data_util import get_decode


def transpose_channels(imgs, label, trajectory_length):
    return tf.transpose(imgs, [0, 3, 1, 2]), label, trajectory_length


def trajectory_cutter(maxlength):
    def cut_off(imgs, label, trajectory_length):
        return imgs[:maxlength], label, tf.minimum(trajectory_length, maxlength)

    return cut_off


def preprocess_dataset(dataset, n_total, n_trajectories_per_batch,
                       trajectory_cutoff_length,
                       x_shape,
                       data_format):
    assert data_format in ['channels_first', 'channels_last']
    dataset = dataset.map(get_decode(x_shape))
    dataset = dataset.map(trajectory_cutter(trajectory_cutoff_length))
    if n_total is not None:
        dataset = dataset.take(n_total)

    if data_format == 'channels_first':
        dataset = dataset.map(transpose_channels)
        batch_shape = (None, x_shape[2], x_shape[0], x_shape[1])
    elif data_format == 'channels_last':
        batch_shape = (None,) + x_shape
    else:
        raise ValueError('Illegal data_format')
    dataset = dataset.padded_batch(n_trajectories_per_batch,
                                   (batch_shape, (), ()))
    return dataset


def get_dataset(filepath, buffer_size, n,
                n_trajectories_per_batch,
                trajectory_cutoff_length,
                x_shape, data_format):
    dataset = tf.data.TFRecordDataset(filenames=[filepath],
                                      compression_type='GZIP',
                                      buffer_size=buffer_size)
    dataset = preprocess_dataset(dataset, n,
                                 n_trajectories_per_batch,
                                 trajectory_cutoff_length,
                                 x_shape,
                                 data_format)
    return dataset


def prepare_and_run_training(train_filepath,
                             valid_filepath,
                             x_shape,
                             n_validation_examples,
                             progress_filepath,
                             model_path,
                             make_predictor,
                             n_trajectories_per_batch,
                             trajectory_cutoff_length,
                             n_epochs_max,
                             task_specific_metrics,
                             n_training_examples=None,
                             trainer_callbacks=None,
                             logger=None,
                             buffer_size=32,
                             data_format='channels_last',
                             debug_mode=False
                             ):
    base_path = os.path.split(progress_filepath)[0]
    step_data_dir = os.path.join(base_path,
                                 'step_data')
    try:
        os.makedirs(step_data_dir)
    except FileExistsError:
        pass
    step_data_filepath = os.path.join(step_data_dir, 'step_data')


    train_dataset = get_dataset(train_filepath, buffer_size,
                                n=None,
                                n_trajectories_per_batch=n_trajectories_per_batch,
                                trajectory_cutoff_length=trajectory_cutoff_length,
                                x_shape=x_shape,
                                data_format=data_format)

    if debug_mode:
        train_dataset = train_dataset.take(2)

    valid_dataset = get_dataset(valid_filepath, buffer_size, n_validation_examples,
                                n_trajectories_per_batch,
                                trajectory_cutoff_length,
                                x_shape, data_format)

    run_training(train_dataset, valid_dataset, make_predictor,
                 n_epochs_max,
                 task_specific_metrics,
                 trainer_callbacks, logger, model_path,
                 progress_filepath, step_data_filepath)



def run_training(train_dataset, valid_dataset, make_predictor, n_epochs_max,
                 task_specific_metrics,
                 trainer_callbacks,
                 logger, model_path, progress_filepath, step_data_filepath):
    data_iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                train_dataset.output_shapes)
    train_iterator_init_op = data_iter.make_initializer(train_dataset)

    valid_iterator_init_op = data_iter.make_initializer(valid_dataset)
    x, y, trajectory_lengths = data_iter.get_next()
    trainer = Trainer(logger=logger)
    predictor = make_predictor(x, trajectory_lengths)
    trainer.run(train_iterator_init_op, valid_iterator_init_op, predictor, n_epochs_max, task_specific_metrics,
                progress_filepath=progress_filepath, step_data_filepath=step_data_filepath,
                callbacks=trainer_callbacks, model_path=model_path)
