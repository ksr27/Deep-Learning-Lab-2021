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
    img_in = np.array(img)
    #hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    rgb = cv2.split(img_in)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    rgb[0] = clahe.apply(rgb[0])
    rgb[1] = clahe.apply(rgb[1])
    rgb[2] = clahe.apply(rgb[2])
    img = cv2.merge(rgb)
    return img#cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

@gin.configurable
@tf.function
def he_tf(img, img_height, img_width):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    r = he_channel(r)
    g = he_channel(g)
    b = he_channel(b)
    #cv2.imwrite("r-he.png", np.array(r))
    img = tf.stack([r, g, b], axis=2)
    #cv2.imwrite("he_img.png", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    return img

@tf.function
def he_channel(img):
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
def preprocess(image, label, img_height, img_width,ds_name):
    """Dataset preprocessing: Normalizing and resizing"""

    # Crop and pad image to square in central region (for idrid dataset)
    if ds_name == "idrid":
        image = tf.image.crop_to_bounding_box(image,offset_height=0,offset_width=270,target_height=2848,target_width=3440)
        image = tf.image.pad_to_bounding_box(image,offset_height=291,offset_width=0,target_height=3440,target_width=3440)

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    image = tf.cast(image, tf.uint8)

    # Normalize image: `uint8` -> `float32`.
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
    if(bool(random.getrandbits(1))):
        image = tf.image.flip_left_right(image)
    random_angle = tf.random.uniform(shape=tf.shape(label),minval=-math.pi, maxval=math.pi)
    image = tfa.image.rotate(image,random_angle,interpolation = 'Bilinear')
    #cropped_image = tf.image.central_crop(rotated_image, central_fraction=0.8)
    if (bool(random.getrandbits(1))):
        image = tf.image.transpose(image)
    #different_rotate = tfa.image.transform_ops.rotate(transposed_image, 1.0472)
    return image, label

