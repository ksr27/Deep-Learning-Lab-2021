## Done by: Lydia

import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import reformat, normalize, balance_augment_ds, get_dataset_moments
from input_pipeline.img_processing import tf_apply_clahe, tf_apply_btg


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

        ds_info = {
            'train': int(ds_info.splits['train'].num_examples * 0.8),
            'val': int(ds_info.splits['train'].num_examples * 0.2),
            'test': ds_info.splits['test'].num_examples,
            'size': (ds_info.splits['train'].num_examples + ds_info.splits['test'].num_examples),
            'num_classes': ds_info.features["label"].num_classes,
            'input_shape': ds_info.features["image"].shape
        }
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "kaggle_dr":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'kaggle_dr',
            split=['train[99%:]', 'train[98%:99%]', 'train[:98%]'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        ds_info = {
            'train': int(ds_info.splits['train'].num_examples * 0.01),
            'val': int(ds_info.splits['train'].num_examples * 0.01),
            'test': int(ds_info.splits['train'].num_examples * 0.98),
            'size': ds_info.splits['train'].num_examples,
            'num_classes': ds_info.features["label"].num_classes,
            'input_shape': ds_info.features["image"].shape
        }
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

        ds_info = {
            'train': ds_info.splits['train'].num_examples,
            'val': ds_info.splits['val'].num_examples,
            'test': ds_info.splits['test'].num_examples,
            'size': (ds_info.splits['train'].num_examples + ds_info.splits['val'].num_examples + ds_info.splits[
                'test'].num_examples),
            'num_classes': ds_info.features["label"].num_classes,
            'input_shape': ds_info.features["image"].shape
        }
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
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, processing_mode):
    # Reformat train images
    ds_train = ds_train.map(reformat, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_info['img_height'] = int(ds_train.element_spec[0].shape[0])
    ds_info['img_width']= int(ds_train.element_spec[0].shape[1])

    # Apply image processing
    if processing_mode == 'clahe':
        ds_train = ds_train.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif processing_mode == 'btg':
        ds_train = ds_train.map(tf_apply_btg, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Get train dataset mean and std for normalization
    ds_mean, ds_std = get_dataset_moments(ds_train)

    # Normalize images to (channelwise) zero mean and unit variance
    ds_train = ds_train.map(lambda img, label: normalize(img, label, ds_mean, ds_std),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()

    # Data augmentation
    ds_train = ds_train.shuffle(ds_info['train'])
    ds_train, ds_info = balance_augment_ds(ds_train, ds_info)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Reformat validation images
    ds_val = ds_val.map(reformat, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply image processing
    if processing_mode == 'clahe':
        ds_val = ds_val.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif processing_mode == 'btg':
        ds_val = ds_val.map(tf_apply_btg, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Normalize images to (channelwise) zero mean and unit variance
    ds_val = ds_val.map(lambda img, label: normalize(img, label, ds_mean, ds_std),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Reformat test images
    ds_test = ds_test.map(reformat, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply image processing
    if processing_mode == 'clahe':
        ds_test = ds_test.map(tf_apply_clahe, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif processing_mode == 'btg':
        ds_test = ds_test.map(tf_apply_btg, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Normalize images to (channelwise) zero mean and unit variance
    ds_test = ds_test.map(lambda img, label: normalize(img, label, ds_mean, ds_std),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_info['batch_size'] = batch_size
    ds_info['train_mean'] = ds_mean
    ds_info['train_std'] = ds_std

    return ds_train, ds_val, ds_test, ds_info
