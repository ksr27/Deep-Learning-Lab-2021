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

