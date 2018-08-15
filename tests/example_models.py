import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv1D, Lambda


def make_example_f(z_shape, seed=1234):
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    with tf.name_scope('dynamics_f'):
        z_1 = Input(batch_shape=[1] + z_shape)
        f_hidden = Conv1D(40,
                          kernel_size=3,
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer)(z_1)
        f_hidden = Conv1D(40, kernel_size=3,
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer)(f_hidden)
        f_hidden = Conv1D(40, kernel_size=3,
                          padding='same',
                          activation='relu',
                          kernel_initializer=initializer)(f_hidden)
        f_out = Conv1D(z_shape[-1], kernel_size=3,
                       padding='same',
                       activation='sigmoid',
                       kernel_initializer=initializer)(f_hidden)
        return Model(inputs=z_1, outputs=f_out, name='f')



def make_identity_model(input_shape):
    model_input = Input(shape=input_shape, name='model_input')
    return Model(inputs=model_input,
                 outputs=model_input,
                 name='identity_model')


def make_constant_keras_model(input_shape, output_tensor):
    """
    "Constant" in the sense of: not depending on input values.
    :param input_shape: Shape of single input
    :param output_value: output value (can be a tensorflow variable!)
    :return: keras model which ignores the input values, but for an input of shape
             ((n,)+input_shape) it outputs a matrix of shape (n,1) where all
             values are output_value/
    """
    inpt = Input(input_shape, name='x')

    def constant_single_output_like(x):
        output_shape = tf.shape(output_tensor)
        return tf.tile(tf.expand_dims(output_tensor, axis=0),
                       multiples=tf.concat([tf.shape(x)[0:1],
                                            tf.ones_like(output_shape)],
                                           axis=0))

    output = Lambda(constant_single_output_like)(inpt)
    return Model(inpt, output)