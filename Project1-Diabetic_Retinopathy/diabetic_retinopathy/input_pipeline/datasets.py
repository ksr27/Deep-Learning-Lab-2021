import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, preprocess2, augment, apply_clahe, he_tf
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
def balance_ds(ds, aug_ds,aug_flag):
    ## returns ds with pos and neg example always following after each other: [0,1,0,1,0,1,...]
    ds_pos = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 1), []))
    if aug_flag:
        ds_neg = ds.filter(lambda image,label: tf.reshape(tf.equal(label, 0), [])) #.repeat()
        aug_ds_pos = aug_ds.filter(lambda image,label: tf.reshape(tf.equal(label, 1), []))
        aug_ds_neg = aug_ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), []))
        ds_neg = ds_neg.concatenate(aug_ds_neg)
        #len_pos = [i for i, _ in enumerate(ds_pos)][-1] + 1
        ds_pos = ds_pos.concatenate(aug_ds_pos.take(39)) #= len(ds_neg)- len(ds_pos) = (330-207)*2-207 #(len(ds)-len_pos)*2-len_pos)
    else:
        ds_neg = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), [])).repeat()
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
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #visualize(ds_train)
    #i=0
    for image,label in ds_train:
        #if(i<10):
        #    cv2.imwrite("./histogramEqualization/4.0-he-clahe/prev_he_img"+str(i)+".png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        image = apply_clahe(image)
        image = tf.convert_to_tensor(image)
        #if (i < 10):
        #    cv2.imwrite("./histogramEqualization/4.0-he-clahe/he_img" + str(i) + ".png",cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        #i = i+1

    # Prepare training dataset
    ds_train = ds_train.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
            ds_train = ds_train.cache()
    # Data augmentation
    ds_train_aug = ds_train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #visualize(ds_train_aug)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = balance_ds(ds_train,ds_train_aug,True)

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image, label in ds_val:
        image = apply_clahe(image)
        image = tf.convert_to_tensor(image)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = balance_ds(ds_val,ds_train_aug,False)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for image, label in ds_test:
        image = apply_clahe(image)
        image = tf.convert_to_tensor(image)
    # Prepare test dataset
    ds_test = ds_test.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = balance_ds(ds_test,ds_train_aug,False)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

