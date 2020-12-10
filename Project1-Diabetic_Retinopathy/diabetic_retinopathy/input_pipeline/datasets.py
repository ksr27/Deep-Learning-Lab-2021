import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess1, preprocess2, augment, histogram_equalization
from input_pipeline.visualize import visualize
import cv2
import numpy as np

def check_distribution(dataset):
    # check class balance
    balance = {'0': 0, '1': 0}
    for (img, label) in dataset:
        if label == 0:
            balance['0'] = balance['0'] + 1
        else:
            balance['1'] = balance['1'] + 1

    balance['perc-0'] = 100 * balance['0'] / (balance['0'] + balance['1'])
    balance['perc-1'] = 100 * balance['1'] / (balance['0'] + balance['1'])
    print("percentage of 0 label:" + str(balance['perc-0']) + "%, percentage of 1 label:" + str(balance['perc-1']))
    return balance

@tf.function
def balance_ds(ds):
    ## returns ds with pos and neg example always following after each other: [0,1,0,1,0,1,...]
    ds_pos = ds.filter(lambda image,label: tf.reshape(tf.equal(label, 1), []))
    ds_neg = ds.filter(lambda image,label: tf.reshape(tf.equal(label, 0), [])).repeat()
    ds = tf.data.Dataset.zip((ds_pos, ds_neg))

    ds = ds.flat_map(
        lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
            tf.data.Dataset.from_tensors(ex_neg)))
    return ds

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
        # train_balance = check_distribution(ds_train)
        # val_balance = check_distribution(ds_val)
        # test_balance = check_distribution(ds_test)
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
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):

    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess1, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for image,label in ds_train:
        #cv2.imwrite("prev_he_img.png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        image = histogram_equalization(image)
        #cv2.imwrite("he_img.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #visualize(ds_train)
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #visualize(ds_train)

    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = balance_ds(ds_train)

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image,label in ds_val:
        image = histogram_equalization(image)
    # Prepare training dataset
    ds_val = ds_val.map(
        preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_val = balance_ds(ds_val)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image,label in ds_test:
        image = histogram_equalization(image)

    ds_test = ds_test.map(
        preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = balance_ds(ds_test)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

