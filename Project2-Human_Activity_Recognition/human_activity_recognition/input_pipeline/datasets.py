import logging
import os.path

import gin
import tensorflow as tf
from input_pipeline.preprocessing import generate_hapt_ds, generate_custom_ds, write_to_file, read_from_file


def check_distribution(dataset, num_classes, mode):
    """check dataset class balance"""
    class_distribution = [0] * num_classes
    len_ds = 0
    for sensor_data, label in dataset:
        if mode == 's2l':
            class_distribution[int(label.numpy())] += 1
            len_ds += 1
        elif mode == 's2s':
            for i in range(num_classes):
                class_distribution[i] += tf.shape(tf.where(label == i))[0].numpy()
            len_ds += tf.shape(sensor_data)[0].numpy()

    return {'class_distribution': class_distribution, 'len': len_ds}


@gin.configurable
def hapt_params(window_length, window_shift, num_classes, mode):
    """get hapt parameters from config.gin"""
    return window_length, window_shift, num_classes, mode


@gin.configurable
def load(name):
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        window_length, window_shift, num_classes, mode = hapt_params()
        if os.path.isfile("./tfrecords/" + mode + "/hapt_train.tfrecord"):
            logging.info(f"Loading dataset from tfrecords...")
            ds_train = read_from_file(split_name="train", window_length=window_length, name=name, mode=mode)
            ds_val = read_from_file(split_name="val", window_length=window_length, name=name, mode=mode)
            ds_test = read_from_file(split_name="test", window_length=window_length, name=name, mode=mode)
        else:
            logging.info(f"Generating dataset from scratch...")
            ds_train = generate_hapt_ds(start_user=1, end_user=21, window_length=window_length,
                                        window_shift=window_shift, mode=mode)
            ds_val = generate_hapt_ds(start_user=28, end_user=30, window_length=window_length,
                                      window_shift=window_shift, mode=mode)
            ds_test = generate_hapt_ds(start_user=22, end_user=27, window_length=window_length,
                                       window_shift=window_shift, mode=mode)
            write_to_file(ds_train, split_name="train", name=name, mode=mode)
            write_to_file(ds_val, split_name="val", name=name, mode=mode)
            write_to_file(ds_test, split_name="test", name=name, mode=mode)

        train_info = check_distribution(ds_train, num_classes, mode=mode)
        val_info = check_distribution(ds_val, num_classes, mode=mode)
        test_info = check_distribution(ds_test, num_classes, mode=mode)

        ds_info = {
            'num_classes': num_classes,
            'missing_classes': False,
            'num_features': 6,
            'label_mapping': ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING',
                              'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                              'LIE_TO_STAND'],
            'sensor_mapping': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            'window_length': window_length,
            'train': train_info,
            'val': val_info,
            'test': test_info,
            'size': train_info['len'] + val_info['len'] + test_info['len'],
            'mode': mode
        }
        return prepare(ds_train, ds_val, ds_test, ds_info)
    elif name == "self_recorded":
        logging.info(f"Preparing dataset {name}...")
        window_length, window_shift, num_classes, mode = hapt_params()
        if os.path.isfile("./tfrecords/" + mode + "/" + name + "_test.tfrecord"):
            logging.info(f"Loading dataset from tfrecords...")
            ds_test = read_from_file(split_name="test", window_length=window_length, name=name, mode=mode)
        else:
            logging.info(f"Generating dataset from scratch...")
            ds_test = generate_custom_ds(window_length=window_length, window_shift=window_shift, mode=mode)
            write_to_file(ds_test, split_name="test", name=name, mode=mode)

        test_info = check_distribution(ds_test, num_classes, mode=mode)

        ds_info = {
            'num_classes': num_classes,
            'missing_classes': True,
            'contained_classes': [1, 2, 3, 4, 5, 6],
            'num_features': 6,
            'label_mapping': ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING',
                              'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                              'LIE_TO_STAND'],
            'sensor_mapping': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            'window_length': window_length,
            'test': test_info,
            'train': test_info,
            'val': test_info,
            'size': test_info['len'],
            'mode': mode
        }
        return prepare(ds_test, ds_test, ds_test, ds_info)
    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    if caching:
        ds_train = ds_train.cache()

    ds_train = ds_train.shuffle(ds_info['train']['len'])
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # validation dataset
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # test dataset
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_info['batch_size'] = batch_size
    return ds_train, ds_val, ds_test, ds_info
