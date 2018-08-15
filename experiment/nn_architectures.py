import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, Lambda
from keras.legacy.layers import merge

import asi.helpers
from asi.helpers import annealing_schedules
from asi.models.asi_model import ASIModel
from utils.tf_util import make_count_variable


def get_f(data_format, architecture, x_shape, args):
    assert data_format in ['channels_first', 'channels_last']

    if data_format == 'channels_first':
        input_shape = (x_shape[2], x_shape[0], x_shape[1])
    elif data_format == 'channels_last':
        input_shape = x_shape
    else:
        raise ValueError('Illegal data_format')

    if architecture == 'f_simple':
        f = get_f_simple(args, data_format, input_shape)
    elif architecture == 'f_dilated':
        f = get_f_dilated(args, data_format, input_shape)
    elif architecture == 'f_strided':
        f = get_f_strided(args, data_format, input_shape, x_shape)
    else:
        raise ValueError('Unknown architecture {}'.format(architecture))
    return f


def get_f_strided(args, data_format, input_shape, x_shape):
    inpt = Input(input_shape)
    hidden = inpt
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    strides=(2, 2),
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2DTranspose(args.n_kernels, (5, 5), padding='same',
                             strides=(2, 2),
                             data_format=data_format,
                             kernel_initializer=args.initializer,
                             activation=args.activation)(hidden)
    hidden = Lambda(lambda x: x[:, :x_shape[0], :x_shape[1], :],
                    # Cut off additional rows
                    output_shape=x_shape[:2] + (args.n_kernels,))(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = merge([inpt, hidden], mode='concat', concat_axis=-1)
    hidden = Conv2D(args.n_kernels, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    output = Conv2D(3, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation='sigmoid')(hidden)
    f = Model(inpt, output, name='f')
    return f


def get_f_dilated(args, data_format, input_shape):
    inpt = Input(input_shape)
    hidden = inpt
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (7, 7), padding='same',
                    dilation_rate=(2, 2),
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = merge([inpt, hidden], mode='concat', concat_axis=-1)
    hidden = Conv2D(args.n_kernels, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    output = Conv2D(3, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation='sigmoid')(hidden)
    f = Model(inpt, output, name='f')
    return f


def get_f_simple(args, data_format, input_shape):
    # Same as D , but without the 1x1-Convolution before the merge
    inpt = Input(input_shape)
    hidden = inpt
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (7, 7), padding='same',
                    dilation_rate=(1, 1),
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = Conv2D(args.n_kernels, (5, 5), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    hidden = merge([inpt, hidden], mode='concat', concat_axis=-1)
    hidden = Conv2D(args.n_kernels, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation=args.activation)(hidden)
    output = Conv2D(3, (1, 1), padding='same',
                    data_format=data_format,
                    kernel_initializer=args.initializer,
                    activation='sigmoid')(hidden)
    f = Model(inpt, output, name='f')
    return f


def make_iterative_predictor(cmd_args, x_shape, x, trajectory_lengths):
    train_step_counter = make_count_variable('train_step_counter', init_value=0)

    f_learning_rate = tf.train.exponential_decay(cmd_args.f_init_learning_rate,
                                                 global_step=train_step_counter,
                                                 decay_steps=cmd_args.f_learning_rate_decay_steps,
                                                 decay_rate=cmd_args.f_learning_rate_decay_rate,
                                                 staircase=True)

    if cmd_args.optimizer == 'adam':
        f_optimizer = tf.train.AdamOptimizer(
            learning_rate=f_learning_rate)
    elif cmd_args.optimizer == 'sgd10':
        f_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=10 * f_learning_rate)

    else:
        raise ValueError('Unknown optimizer: {}'.format(cmd_args.optimizer))

    f = get_f(cmd_args.data_format, cmd_args.f_architecture, x_shape, cmd_args)

    if cmd_args.z_loss_fn == 'log_loss':
        z_loss_fn = asi.helpers.z_loss_fns.log_loss
    else:
        raise ValueError('Unknown z_loss_fn {}'.format(cmd_args.z_loss_fn))

    if cmd_args.data_format == 'channels_first':
        input_shape = (x_shape[2], x_shape[0], x_shape[1])
    elif cmd_args.data_format == 'channels_last':
        input_shape = x_shape
    else:
        raise ValueError('Illegal data format')

    additional_metrics = {'f_learning_rate': f_learning_rate, }
    schd_sampling_schedule = annealing_schedules.get_reciprocal(
        decay=1. / (0.2 * cmd_args.schd_sampling_steps))
    exploration_schedule = annealing_schedules.get_linear(
        cmd_args.exploration_steps, 1.0, 0)
    latent_predictor = ASIModel(
        x_shape=input_shape,
        f_optimizer=f_optimizer,
        f=f,
        delta_t_bounds=(cmd_args.delta_t_lower_bound, cmd_args.delta_t_upper_bound),
        exploration_schedule=exploration_schedule,
        schd_sampling_schedule=schd_sampling_schedule,
        additional_metrics=additional_metrics,
        x=x,
        trajectory_lengths=trajectory_lengths,
        parallel_iterations=10,
        train_step_counter=train_step_counter,
        z_loss_fn=z_loss_fn,
    )
    return latent_predictor