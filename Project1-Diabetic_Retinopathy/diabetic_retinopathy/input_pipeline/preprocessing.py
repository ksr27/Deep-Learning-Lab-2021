import gin
import tensorflow as tf
import tensorflow_addons as tfa
import math as math
import random


@tf.function
@gin.configurable
def reformat(image, label, img_height, img_width, ds_name):
    """Dataset preprocessing: Cropping, padding and resizing"""

    # Crop and pad image to square in central region
    if ds_name == "idrid":
        image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=270, target_height=2848,
                                              target_width=3440)
        image = tf.image.pad_to_bounding_box(image, offset_height=291, offset_width=0, target_height=3440,
                                             target_width=3440)
    elif ds_name == "kaggle_dr":
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        if (height < width):
            image = tf.image.pad_to_bounding_box(image, offset_height=int((width - height) / 2), offset_width=0,
                                                 target_height=width, target_width=width)
    elif ds_name == "eyepacs":
        if label < 2:
            label = 0
        else:
            label = 1

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    image = tf.cast(image, tf.uint8)
    return image, label


@gin.configurable
def get_dataset_moments(ds):
    """Compute dataset R,G,B-channel-mean and std
    """
    mean = tf.zeros(shape=(3,))
    std = tf.zeros(shape=(3,))

    for image, label in ds:
        image = tf.cast(image, tf.float32) / 255.
        mean += tf.math.reduce_mean(image, axis=(0, 1))
        std += tf.math.reduce_std(image, axis=(0, 1))

    mean = mean / len(ds)
    std = std / len(ds)

    return mean, std


@tf.function
def normalize(image, label, ds_mean, ds_std):
    """Dataset preprocessing: Normalizing:
        1. img/255 -> [0,1] scale
        2. channelwise zero-mean unit variance: img-channel_mean/(channel_std).
    """

    image = tf.cast(image, tf.float32) / 255.
    image = (image - ds_mean) / ds_std

    return image, label


@tf.function
def augment(image, label): #Baran
    """(randomized) Data augmentation
        * flip image
        * rotate image at random angle
        * transpose image
    """

    # flip image
    if (bool(random.getrandbits(1))):
        image = tf.image.flip_left_right(image)

    # rotate image at random angle
    random_angle = tf.random.uniform(shape=tf.shape(label), minval=-math.pi, maxval=math.pi)
    image = tfa.image.rotate(image, random_angle, interpolation='Bilinear')

    # transpose image
    if (bool(random.getrandbits(1))):
        image = tf.image.transpose(image)
    return image, label


def check_distribution(dataset):
    """ analyze class distribution in dataset
    """
    balance = {'0': 0, '1': 0}
    for img, label in dataset:
        if label == 0:
            balance['0'] += 1
        else:
            balance['1'] += 1
    return balance


@gin.configurable
def balance_augment_ds(ds, ds_info, aug_perc):
    """balances ds to equal amount of 0 and 1 labels by either augmenting or repeating

    Parameters:
        ds_info: dict containing dataset information
        ds (tf.data.Dataset): dataset to balance
        aug_perc (float): percentage of ds to be augmented

    Returns:
        ds (tf.data.Dataset): balanced dataset
    """
    # returns ds with pos and neg example always following after each other: [0,1,0,1,0,1,...]
    ds_pos = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 1), []))
    ds_neg = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), []))

    # augment specified percentage of train dataset
    aug_ds = ds.repeat().take(int(len(ds) * aug_perc))
    aug_ds = aug_ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # check amount of '0' and '1' samples in original and augmented dataset
    balance_aug = check_distribution(aug_ds)
    balance_dataset = check_distribution(ds)

    # returns ds with pos and neg example always following after each other: [0,1,0,1,0,1,...]
    aug_ds_pos = aug_ds.filter(lambda image, label: tf.reshape(tf.equal(label, 1), []))
    aug_ds_neg = aug_ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), []))
    ds_neg = ds_neg.concatenate(aug_ds_neg)

    # computes difference between '0' samples (original dataset+augmented) and '1' samples (original dataset)
    num_take = balance_dataset['0'] + balance_aug['0'] - balance_dataset['1']
    if num_take < 0:  # not enough additional samples to equal imbalance by augmenting
        ds_neg = ds_neg.repeat()
        ds_info['size'] = 2 * balance_dataset['1']
    elif num_take > 0:  # more than enough samples, equal previous imbalance + take additional augmented '1' samples
        ds_pos = ds_pos.concatenate(aug_ds_pos.take(num_take))
        ds_info['size'] = (balance_dataset['0'] + balance_aug['0']) * 2
    else:  # num_take == 0
        ds_info['size'] = 2 * balance_dataset['1']

    # add '0' and '1' samples back together
    ds = tf.data.Dataset.zip((ds_pos, ds_neg))
    ds = ds.flat_map(
        lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
            tf.data.Dataset.from_tensors(ex_neg)))
    return ds, ds_info
