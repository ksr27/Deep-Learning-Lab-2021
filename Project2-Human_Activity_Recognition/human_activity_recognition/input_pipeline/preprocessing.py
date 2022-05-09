import glob

import gin
import numpy as np
import pandas as pd
import tensorflow as tf


def take_prominent_label(sensor_data, label):
    label_name, __, label_count = tf.unique_with_counts(label)
    label = label_name[tf.math.argmax(label_count)]
    return sensor_data, label


@gin.configurable
def generate_hapt_ds(start_user, end_user, window_length, window_shift, mode, path):
    """Generates hapt dataset from files on path with specified start and end user, samples get split into windows with
    window_length and window_shift.

    Parameters:
        start_user (int): starting from this user data will be taken for the dataset
        end_user (int): last user to take data for dataset from
        window_length (int): single samples will be grouped into windows of size window_length
        window_shift (int): windowing will be done with a shift length of window_shift
        path (str): path to the raw user data

    Returns:
        dataset (tf.tf.data.Dataset): generated and windowed Dataset
    """
    labelLine = [line.rstrip('\n') for line in open(path + '/labels.txt')]
    result = []
    label = []

    for i in range(start_user, end_user + 1):
        user_label = []
        user_result = []

        if i < 10:
            acc_glob = path + '/acc_exp*_user0' + str(i) + '.txt'
            gyro_glob = path + '/gyro_exp*_user0' + str(i) + '.txt'
        else:
            acc_glob = path + '/acc_exp*_user' + str(i) + '.txt'
            gyro_glob = path + '/gyro_exp*_user' + str(i) + '.txt'

        acc_paths = glob.glob(acc_glob)
        gyro_paths = glob.glob(gyro_glob)
        acc_lines = {}
        gyro_lines = {}
        for file in acc_paths:
            experiment_id = int(file[-13:-11])
            acc_lines[experiment_id] = [line.rstrip('\n') for line in open(file)]

        for file in gyro_paths:
            experiment_id = int(file[-13:-11])
            gyro_lines[experiment_id] = [line.rstrip('\n') for line in open(file)]

        # Column 1: experiment number ID,
        # Column 2: user number ID,
        # Column 3: activity number ID
        # Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
        # Column 5: Label end point (in number of signal log samples)
        prev_end = 0
        for entries in labelLine:
            entries = entries.split(' ')
            experiment_id = int(entries[0])
            user_id = int(entries[1])
            activity = int(entries[2])
            start_point = int(entries[3])
            end_point = int(entries[4])
            if user_id == i:
                acc1 = []
                acc2 = []
                acc3 = []
                gyro1 = []
                gyro2 = []
                gyro3 = []
                experiment_label = []
                experiment_result = []

                if prev_end > start_point:
                    prev_end = 0
                if start_point > prev_end + 1:
                    acc_line = acc_lines[experiment_id]
                    gyro_line = gyro_lines[experiment_id]
                    for j in range(prev_end, start_point - 1):
                        acc_sensors = acc_line[j].split(' ')
                        gyro_sensors = gyro_line[j].split(' ')
                        acc1.append(float(acc_sensors[0]))
                        acc2.append(float(acc_sensors[1]))
                        acc3.append(float(acc_sensors[2]))
                        gyro1.append(float(gyro_sensors[0]))
                        gyro2.append(float(gyro_sensors[1]))
                        gyro3.append(float(gyro_sensors[2]))

                    experiment_label = np.zeros_like(acc1, dtype=int)

                acc_line = acc_lines[experiment_id]
                gyro_line = gyro_lines[experiment_id]

                for j in range(start_point, end_point + 1):
                    acc_sensors = acc_line[j].split(' ')
                    gyro_sensors = gyro_line[j].split(' ')
                    acc1.append(float(acc_sensors[0]))
                    acc2.append(float(acc_sensors[1]))
                    acc3.append(float(acc_sensors[2]))
                    gyro1.append(float(gyro_sensors[0]))
                    gyro2.append(float(gyro_sensors[1]))
                    gyro3.append(float(gyro_sensors[2]))

                acc1 = np.array(acc1)
                acc2 = np.array(acc2)
                acc3 = np.array(acc3)
                gyro1 = np.array(gyro1)
                gyro2 = np.array(gyro2)
                gyro3 = np.array(gyro3)

                if len(experiment_label) == 0:
                    experiment_label = np.ones(len(acc1), dtype=int) * activity
                else:
                    experiment_label = np.concatenate(
                        (experiment_label, np.ones(len(acc1) - len(experiment_label), dtype=int) * activity), axis=0)

                prev_end = end_point
                experiment_result = np.column_stack((acc1, acc2, acc3, gyro1, gyro2, gyro3))

                if len(user_result) == 0:
                    user_result = experiment_result
                    user_label = experiment_label
                else:
                    user_result = np.concatenate((user_result, experiment_result), axis=0)
                    user_label = np.concatenate((user_label, experiment_label), axis=0)

        # normalization for each user
        user_result = (user_result - np.mean(user_result, axis=0)) / np.std(user_result, axis=0)
        if len(result) == 0:

            result = user_result
            label = user_label
        else:
            result = np.concatenate((result, user_result), axis=0)
            label = np.concatenate((label, user_label), axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((result, label))
    dataset = dataset.window(size=window_length, shift=window_shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(window_length), y.batch(window_length))))

    if mode == 's2l':
        dataset = dataset.map(take_prominent_label)
    return dataset


@gin.configurable
def generate_custom_ds(window_length, window_shift, mode, path):
    """Generates self-recorded dataset from csv files on path, samples get split into windows with window_length
    and window_shift.

    Parameters:
        window_length (int): single samples will be grouped into windows of size window_length
        window_shift (int): windowing will be done with a shift length of window_shift
        mode (str): store samples for s2s or s2l classifcation
        path (str): path to the raw user data

    Returns:
        dataset (tf.tf.data.Dataset): generated and windowed Dataset
    """

    acc_paths = []
    gyro_paths = []
    result = []
    labels = []

    label_lookup = ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING',
                    'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                    'LIE_TO_STAND']

    csv_paths = glob.glob(path + '/*.csv')
    csv_paths.sort()
    for path in csv_paths:
        if 'Accelerometer' in path:
            acc_paths.append(path)
        else:
            gyro_paths.append(path)

    for i in range(len(acc_paths)):
        acc = pd.read_csv(acc_paths[i], names=['timestamp', 'x', 'y', 'z', 'label'])
        gyro = pd.read_csv(gyro_paths[i], names=['timestamp', 'x', 'y', 'z', 'label'])
        data_label = acc['label'][0]

        acc = np.column_stack((acc['x'].to_numpy(), acc['y'].to_numpy(), acc['z'].to_numpy()))
        gyro = np.column_stack((gyro['x'].to_numpy(), gyro['y'].to_numpy(), gyro['z'].to_numpy()))

        # in case of uneven number of acc and gyro data points
        diff = acc.shape[0] - gyro.shape[0]
        if diff > 0:
            acc = acc[:gyro.shape[0], :]
        elif diff < 0:
            gyro = gyro[:acc.shape[0], :]

        result.append(np.column_stack((acc, gyro)))

        for j, label in enumerate(label_lookup):
            if str.lower(label) == str.lower(data_label):
                labels.append(np.ones(acc.shape[0], dtype=int) * j)

    # get channel mean over all experiments
    mean = np.mean(np.vstack(result), axis=0)
    std = np.std(np.vstack(result), axis=0)

    for i, single_exp in enumerate(result):
        single_exp = (single_exp - mean) / std

        single_ds = tf.data.Dataset.from_tensor_slices((single_exp, labels[i]))
        single_ds = single_ds.window(size=window_length, shift=window_shift, drop_remainder=True)
        single_ds = single_ds.flat_map(
            lambda x, y: tf.data.Dataset.zip((x.batch(window_length), y.batch(window_length))))

        if (i == 0):
            dataset = single_ds
        else:
            dataset = dataset.concatenate(single_ds)

    if mode == 's2l':
        dataset = dataset.map(take_prominent_label)
    return dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_to_file(dataset, split_name, name, mode):
    """Write dataset to TFRecord file(s)

    Parameters:
        dataset (tf.tf.data.Dataset): Dataset to write to TFRecord file
        split_name (str): which split to get (train, val or test)

    Returns:
        Nothing
    """
    if mode == 's2l':
        def serialize_example(sensor_data, label):
            """
            Creates a tf.train.Example message ready to be written to a file.
            """
            feature = {
                'sensor': _bytes_feature(tf.io.serialize_tensor(sensor_data)),
                'label': _int64_feature(label),
            }

            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

    elif mode == 's2s':
        def serialize_example(sensor_data, label):
            """
            Creates a tf.train.Example message ready to be written to a file.
            """
            feature = {
                'sensor': _bytes_feature(tf.io.serialize_tensor(sensor_data)),
                'label': _bytes_feature(tf.io.serialize_tensor(label)),
            }

            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

    def tf_serialize_example(sensor_data, label):
        tf_string = tf.py_function(
            serialize_example,
            (sensor_data, label),
            tf.string)
        return tf.reshape(tf_string, ())

    serialized_dataset = dataset.map(tf_serialize_example)

    filename = 'tfrecords/' + mode + '/' + name + '_' + split_name + '.tfrecord'

    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_dataset)


@gin.configurable
def read_from_file(split_name, window_length, name, mode):
    """Get dataset from TFRecord file(s)

    Parameters:
        split_name (str): which split to get (train, val or test)
        window_length (int): window_length of the samples in the dataset

    Returns:
        dataset (tf.tf.data.Dataset): extracted Dataset
    """
    filename = 'tfrecords/' + mode + '/' + name + '_' + split_name + '.tfrecord'
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    # Create a description of the features.
    if mode == 's2l':
        feature_description = {
            'sensor': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            sensor_data = tf.io.parse_tensor(example['sensor'], out_type=tf.float64)
            sensor_data = tf.reshape(sensor_data, [window_length, 6])
            label = example['label']
            return sensor_data, label

    elif mode == 's2s':
        feature_description = {
            'sensor': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value='0'),
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            example = tf.io.parse_single_example(example_proto, feature_description)
            sensor_data = tf.io.parse_tensor(example['sensor'], out_type=tf.float64)
            sensor_data = tf.reshape(sensor_data, [window_length, 6])
            label = tf.io.parse_tensor(example['label'], out_type=tf.int64)
            return sensor_data, label

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
