import matplotlib

from asi.callbacks.batch_notification_callback import BatchNotificationCallback
from experiment import task_accuracy_fns
from experiment.nn_architectures import make_iterative_predictor

matplotlib.use('Agg')

from asi.callbacks.timeline_callback import TimelineCallback
from asi.callbacks.saver_callback import SaverCallback

import argparse
import json
import os
from functools import partial

from asi.train_dispatcher import prepare_and_run_training
from asi.callbacks.video_callback import PredictionVideoCallback
from utils.data_util import get_batch_from_tfrecords_dataset
from utils.misc import get_nice_logger, ObjectSaver


def main(args, logger):
    if args.debug_mode:
        for i in range(5):
            logger.info('=' * 80)
            logger.info('== D E B U G  M O D E ==')

    logger.warning('The random_seed is currently ignored.')

    n_training_examples = 500

    make_predictor = make_iterative_predictor

    models_dir = os.path.join(args.output_dir, 'models')
    try:
        os.makedirs(models_dir)
    except FileExistsError:
        pass

    trainer_callbacks = [
        BatchNotificationCallback(logger),
    ]

    video_batch_valid, video_callback_valid = make_video_callback(args, logger,
                                                                  name='valid',
                                                                  filepath=args.valid_filepath)
    _, video_callback_train = make_video_callback(args, logger,
                                                  name='train',
                                                  filepath=args.train_filepath)
    timeline_callback = make_timeline_callback(args, logger, video_batch_valid)
    object_saver, object_saver_callback = make_object_saver_callback(args)

    trainer_callbacks.extend([
        timeline_callback,
        video_callback_valid,
        video_callback_train,
        object_saver_callback,
    ])

    if args.dataset == 'fubo':
        task_specific_metrics = {
            'task_specific_fair_accuracy':
                task_accuracy_fns.fubo_fair_accuracy,
            'object_maintenance_frames':
                task_accuracy_fns.fubo_ball_maintenance_frames,
        }
    elif args.dataset == 'rr':
        task_specific_metrics = {'task_specific_fair_accuracy':
                                     task_accuracy_fns.rr_fair_accuracy,
                                 'object_maintenance_frames':
                                     task_accuracy_fns.rr_runner_maintenance_frames,
                                 }
    else:
        raise NotImplementedError('Please implement task-specific metrics for dataset {}'.format(args.dataset))

    prepare_and_run_training(
        args.train_filepath,
        args.valid_filepath,
        x_shape=X_SHAPE,
        n_validation_examples=args.n_validation_examples,
        model_path=os.path.join(models_dir, 'model.chkpt'),
        progress_filepath=os.path.join(args.output_dir, 'progress.dat'),
        make_predictor=partial(make_predictor, args, X_SHAPE),
        n_trajectories_per_batch=args.n_trajectories_per_batch,
        trajectory_cutoff_length=185,
        n_epochs_max=args.n_epochs,
        task_specific_metrics=task_specific_metrics,
        n_training_examples=n_training_examples,
        trainer_callbacks=trainer_callbacks,
        logger=logger,
        data_format=args.data_format,
        debug_mode=args.debug_mode
    )


def make_object_saver_callback(args):
    object_saver = ObjectSaver(os.path.join(args.output_dir,
                                            'saved_objects.pkl'))
    saver_callback = SaverCallback(object_saver)
    return object_saver, saver_callback


def make_timeline_callback(args, logger, video_batch):
    timeline_dir = os.path.join(args.output_dir, 'timelines')
    try:
        os.makedirs(timeline_dir)
    except FileExistsError:
        pass
    n_timelines = 6

    if args.debug_mode:
        n_timelines = 1

    timeline_callback = TimelineCallback(logger, timeline_dir, video_batch[:n_timelines],
                                         every_n_epochs=5,
                                         delta_t_bounds=(args.delta_t_lower_bound,
                                                         args.delta_t_upper_bound),
                                         variations=['C'],
                                         paper_mode=False,
                                         )
    return timeline_callback


def make_video_callback(args, logger, name, filepath):
    video_dir = os.path.join(args.output_dir, 'videos_{}'.format(name))
    try:
        os.makedirs(video_dir)
    except FileExistsError:
        pass
    n_trajectories_per_video = 5
    n_videos = 4

    if args.debug_mode:
        n_videos = 1

    video_batch, _, _ = get_batch_from_tfrecords_dataset(
        filepath,
        n_examples=n_trajectories_per_video * n_videos,
        x_shape=X_SHAPE)
    video_callback = PredictionVideoCallback(logger, video_dir, video_batch,
                                             fps=30,
                                             every_n_epochs=5,
                                             scale_factor=2,
                                             predictor_data_format=args.data_format,
                                             show_z_only=True,
                                             show_bar_charts=False,
                                             n_videos=n_videos)
    return video_batch, video_callback


def dump_args(args: dict, directory):
    filepath = os.path.join(directory, 'args.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing_args = json.load(f)
            if existing_args != args:
                raise ValueError('The output directory already contains a run with '
                                 'different arguments!')

    with open(filepath, 'w') as f:
        json.dump(args, f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--optimizer', type=str, required=True,
                           help='Optimizer used for asi')
    argparser.add_argument('--f_init_learning_rate', type=float, required=True,
                           help='Learning rate for the f Model')
    argparser.add_argument('--f_learning_rate_decay_rate', type=float, required=True,
                           help='Learning rate decay rate')
    argparser.add_argument('--f_learning_rate_decay_steps', type=float, required=True,
                           help='Learning rate decay steps')
    argparser.add_argument('--z_loss_fn', type=str, required=False, default=None,
                           help='One of "log_loss" and "symmetric_log_loss"')
    argparser.add_argument('--output_dir', type=str, required=True,
                           help='Path where to save the results')
    argparser.add_argument('--n_epochs', type=int, required=True,
                           help='Number of epochs to train the model')
    argparser.add_argument('--n_trajectories_per_batch', type=int, required=True,
                           help='Batch size of the asi procedure')
    argparser.add_argument('--n_validation_examples', type=int, default=500,
                           help='Number of examples to use for validation')
    argparser.add_argument('--exploration_steps', type=int, default=None,
                           help='Number of asi steps to anneal exploration '
                                'temperature')
    argparser.add_argument('--schd_sampling_steps', type=int, default=None,
                           help='Number of asi steps to anneal scheduled sampling '
                                'temperature')
    argparser.add_argument('--f_architecture', type=str, default=None,
                           help='Variation of the architecture of f')
    argparser.add_argument('--delta_t_upper_bound', type=int, default=None,
                           help='Upper bound for t-jumps')
    argparser.add_argument('--delta_t_lower_bound', type=int, default=1,
                           help='Lower bound for t-jumps')
    argparser.add_argument('--activation', type=str, required=True,
                           choices=['relu', 'elu'],
                           help='Activation function for all networks')
    argparser.add_argument('--n_kernels', type=int, required=True,
                           help='Number of convolutional kernels for f network')
    argparser.add_argument('--initializer', type=str, required=True,
                           choices=['he_uniform', 'glorot_uniform'],
                           help='Weight initializers for all networks')
    argparser.add_argument('--dataset', type=str, required=True,
                           choices=['fubo', 'rr'])

    argparser.add_argument('--train_filepath', type=str, required=True)
    argparser.add_argument('--valid_filepath', type=str, required=True)

    argparser.add_argument('--data_format', type=str, default='channels_last',
                           help='Conv2D Data Format.')
    argparser.add_argument('--debug_mode', dest='debug_mode', action='store_true')
    argparser.set_defaults(debug_mode=False)
    args = argparser.parse_args()

    logger_output_file = os.path.join(args.output_dir, 'log_output.log')
    logger = get_nice_logger(__file__, logger_output_file, file_exists_behavior='append')

    logger.info('##################################################')
    logger.info('###  START RUN  ###')
    logger.info('##################################################')

    logger.info('Arguments:')
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    logger.info('Writing arguments...')
    dump_args(vars(args), args.output_dir)

    if args.dataset == 'fubo':
        X_SHAPE = (126, 75, 3)

    elif args.dataset == 'rr':
        X_SHAPE = (64, 64, 3)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    logger.info('Starting Training...')
    main(args, logger)
