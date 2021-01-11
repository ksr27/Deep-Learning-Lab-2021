import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, augment

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

@gin.configurable
def load(name, data_dir):
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'hapt',
            split=['train', 'val', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        return prepare(ds_train, ds_val, ds_test, ds_info)
    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, window_length, window_shift):

    # Prepare training dataset
    #ds_train = ds_train.map(
    #    preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.window(size=window_length,shift=window_shift, drop_remainder=True)

    for sensor_data,label in ds_train:
        print(sensor_data,label)
    if caching:
            ds_train = ds_train.cache()
    # Data augmentation
    #ds_train_aug = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

