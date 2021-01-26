import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, preprocess2, augment, apply_clahe, balance_ds, get_dataset_moments, tf_apply_clahe
from input_pipeline.visualize import visualize
import cv2
import numpy as np


@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'idrid',
            split=['train[:80%]', 'train[80%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "kaggle_dr":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'kaggle_dr',
            split=['train[90%:]', 'train[80%:90%]', 'train[:80%]'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, clahe_flag):

    ds_mean, ds_std = get_dataset_moments(ds_train)

    # Prepare training dataset
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #visualize(ds_train)
    if clahe_flag:
        ds_train = ds_train.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prepare training dataset
    #ds_train = ds_train.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.map(lambda img,label: preprocess2(img,label,ds_mean,ds_std),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()

    # Data augmentation
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train, ds_train_size = balance_ds(ds_train,'train')

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset part 1
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if clahe_flag:
        ds_val = ds_val.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset part 2
    ds_val = ds_val.map(lambda img,label: preprocess2(img,label,ds_mean,ds_std),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_val = ds_val.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_val,val_size = balance_ds(ds_val,'val')
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if clahe_flag:
        ds_test = ds_test.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
    ds_test = ds_test.map(lambda img,label: preprocess2(img,label,ds_mean,ds_std),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_test = ds_test.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds_test,test_size = balance_ds(ds_test,'test')
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info, ds_train_size, batch_size
