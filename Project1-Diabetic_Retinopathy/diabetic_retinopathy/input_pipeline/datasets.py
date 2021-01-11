import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, preprocess2, augment, apply_clahe, balance_ds
from input_pipeline.visualize import visualize
import cv2


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

    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #visualize(ds_train)
    #i=0
    if clahe_flag:
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

    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = balance_ds(ds_train,'train')

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if clahe_flag:
        for image, label in ds_val:
            image = apply_clahe(image)
            image = tf.convert_to_tensor(image)

    # Prepare validation dataset
    ds_val = ds_val.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = balance_ds(ds_val,'val')
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if clahe_flag:
        for image, label in ds_test:
            image = apply_clahe(image)
            image = tf.convert_to_tensor(image)
            
    ds_test = ds_test.map(preprocess2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = balance_ds(ds_test,'test')
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
