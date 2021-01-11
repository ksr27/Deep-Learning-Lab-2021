import gin
import tensorflow as tf
import numpy as np
import cv2
import tensorflow_addons as tfa
import math as math
import random
#import mclahe as mc

@gin.configurable
def apply_clahe(img,clip_limit):
    """applies contrast limited adaptive histogram equalization to image

    Parameters:
        img (int,int,int): input image (f.e. (256,256,3))
        clip_limit (string): limit for amplification by clipping the histogram at this value

    Returns:
        img (int,int): image with clahe applied to it
    """

    img_in = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    # convert img to hsv (hue, saturation, value) format
    hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    hsv = cv2.split(hsv)

    # apply clahe to value channel only
    hsv[2] = clahe.apply(hsv[2])

    hsv = cv2.merge(hsv)
    #rgb[0] = clahe.apply(rgb[0])
    #rgb[1] = clahe.apply(rgb[1])
    #rgb[2] = clahe.apply(rgb[2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

@gin.configurable
@tf.function
def he_tf(img, img_height, img_width):
    """applies histogram equalization to image

    Parameters:
        img (int,int,int): input image
        img_height (int): image height
        img_width (int): image width

    Returns:
        img (int,int): histogram equalized image
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # run he for each channel separately
    r = he_channel(r)
    g = he_channel(g)
    b = he_channel(b)
    img = tf.stack([r, g, b], axis=2)
    return img

@tf.function
def he_channel(img):
    """applies histogram equalization to one channel

    Parameters:
        img (int,int): one channel input image

    Returns:
        img (int,int): histogram equalized one channel image
    """
    img = tf.expand_dims(img,axis = 2)
    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(img,tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(img)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min,tf.float32) * 255. / tf.cast(pix_cnt - 1,tf.float32))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.gather_nd(px_map, tf.cast(img, tf.int32))
    #clahe = mc.mclahe(eq_hist, kernel_size=(8,8), clip_limit = 8.0)
    return eq_hist

@tf.function
@gin.configurable
def preprocess(image, label, img_height, img_width,ds_name, he_flag):
    """Dataset preprocessing: Normalizing and resizing"""

    # Crop and pad image to square in central region (for idrid dataset)
    if ds_name == "idrid":
        image = tf.image.crop_to_bounding_box(image,offset_height=0,offset_width=270,target_height=2848,target_width=3440)
        image = tf.image.pad_to_bounding_box(image,offset_height=291,offset_width=0,target_height=3440,target_width=3440)

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    image = tf.cast(image, tf.uint8)

    # Normalize image: `uint8` -> `float32`.
    if he_flag:
        image = he_tf(image)
    #image = tf.cast(image, tf.float32)/255.

    return image, label

@tf.function
@gin.configurable
def preprocess2(image, label):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32)/255.

    return image, label

@tf.function
def augment(image, label):
    """Data augmentation"""

    # flip image
    if(bool(random.getrandbits(1))):
        image = tf.image.flip_left_right(image)

    # rotate iamge at random angle
    random_angle = tf.random.uniform(shape=tf.shape(label),minval=-math.pi, maxval=math.pi)
    image = tfa.image.rotate(image,random_angle,interpolation = 'Bilinear')

    # transpose image
    if (bool(random.getrandbits(1))):
        image = tf.image.transpose(image)
    return image, label

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
    #print("percentage of 0 label:" + str(balance['perc-0']) + "%, percentage of 1 label:" + str(balance['perc-1']))
    return balance

@tf.function
@gin.configurable
def balance_ds(ds,split,aug_perc):
    """balances ds to equal amount of 0 and 1 labels by either augmenting or repeating

    Parameters:
        ds (tf.data.Dataset): datasets to balance
        split (string): name of ds
        aug_perc (float): percentage of ds to be augmented

    Returns:
        ds (tf.data.Dataset): balanced dataset
    """
    ## returns ds with pos and neg example always following after each other: [0,1,0,1,0,1,...]
    ds_pos = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 1), []))
    ds_neg = ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), []))
    if split == 'train':
        # augment specified percentage of train dataset
        aug_ds = ds.repeat().take(int(len(ds)*aug_perc))
        aug_ds = aug_ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        balance_aug = check_distribution(aug_ds)
        balance_ds = check_distribution(ds)

        aug_ds_pos = aug_ds.filter(lambda image, label: tf.reshape(tf.equal(label, 1), []))
        aug_ds_neg = aug_ds.filter(lambda image, label: tf.reshape(tf.equal(label, 0), []))
        
        ds_neg = ds_neg.concatenate(aug_ds_neg)
        num_take = balance_ds['0']+balance_aug['0']-balance_ds['1']
        ds_pos = ds_pos.concatenate(aug_ds_pos.take(num_take)) #= len(ds_neg)- len(ds_pos) = (330-207)*2-207 #(len(ds)-len_pos)*2-len_pos)
    else:
        ds_neg = ds_neg.repeat()
    ds = tf.data.Dataset.zip((ds_pos, ds_neg))
    ds = ds.flat_map(
        lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
            tf.data.Dataset.from_tensors(ex_neg)))
    return ds
