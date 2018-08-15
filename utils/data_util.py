import tensorflow as tf


def get_batch_from_tfrecords_dataset(filepath, n_examples, x_shape):
    dataset = tf.data.TFRecordDataset(filenames=[filepath],
                                      compression_type='GZIP', buffer_size=0)
    dataset = dataset.map(get_decode(x_shape))
    dataset = dataset.take(n_examples)
    iterator = dataset.make_one_shot_iterator()
    get_next_op = iterator.get_next()
    init_op = iterator.make_initializer(dataset)
    x_batch = []
    labels = []
    trajectory_lengths = []
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(n_examples):
            x, label, trajectory_length = sess.run(get_next_op)
            x_batch.append(x)
            labels.append(label)
            trajectory_lengths.append(trajectory_length)
    return x_batch, labels, trajectory_lengths


def get_decode(x_shape):
    def decode(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={'x': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.int64),
                      'trajectory_length': tf.FixedLenFeature([], tf.int64)
                      })

        trajectory_length = tf.cast(features['trajectory_length'], tf.int32)
        images_raw = tf.decode_raw(features['x'], tf.uint8)
        trajectory_shape = tf.concat(((trajectory_length,), x_shape), axis=0)
        x = tf.reshape(images_raw, trajectory_shape)
        x = tf.cast(x, tf.float32, name='x')
        x = x / 255.
        label = tf.cast(features['label'], tf.int32, name='label')
        return x, label, trajectory_length

    return decode
