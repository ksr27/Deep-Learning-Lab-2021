import gin
import tensorflow as tf
import numpy as np
import cv2


@gin.configurable
def apply_clahe(image, label, clip_limit):
    """applies contrast limited adaptive histogram equalization to image

    Parameters:
        image (int,int,int): input image (f.e. (256,256,3))
        clip_limit (string): limit for amplification by clipping the histogram at this value

    Returns:
        img (int,int): image with clahe applied to it
    """

    img_in = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    # convert img to hsv (hue, saturation, value) format
    hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    hsv = cv2.split(hsv)

    # apply clahe to value channel only
    hsv[2] = clahe.apply(hsv[2])

    hsv = cv2.merge(hsv)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hsv = tf.convert_to_tensor(hsv, dtype=tf.uint8)
    return hsv, label


def tf_apply_clahe(image, label):
    """py function wrapper for clahe"""

    [image, label] = tf.py_function(apply_clahe, [image, label], [tf.uint8, tf.int64])
    label.set_shape([])
    image.set_shape([None, None, 3])
    return image, label


@gin.configurable
def apply_btg(image, label, sigmaX=10):
    """applies an adaption of ben graham preprocessing to image

    Parameters:
        image (int,int,int): input image (f.e. (256,256,3))
        sigmaX (string): variance value for Gaussian smoothing

    Returns:
        img (int,int): image with ben graham preprocessing applied to it
    """

    image = np.array(image)

    # get local averages by smoothing the image with a gaussian
    gaussian_smoothed = cv2.GaussianBlur(image, (0, 0), sigmaX)

    # subtract local average from original image, map 0 values to grey (=128 for all channels)
    image = cv2.addWeighted(image, 4, gaussian_smoothed, -4, 128)
    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    return image, label


def tf_apply_btg(image, label):
    """py function wrapper for btg"""
    [image, label] = tf.py_function(apply_btg, [image, label], [tf.uint8, tf.int64])
    label.set_shape([])
    image.set_shape([None, None, 3])
    return image, label
