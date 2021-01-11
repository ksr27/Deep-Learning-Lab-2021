import gin
import tensorflow as tf
import tensorflow_addons as tfa
import math as math
import random
import tensorflow_transform as tft

@tf.function
@gin.configurable
def preprocess(sensor_data,label):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize each channel
    mean, var = tf.nn.moments(sensor_data, axes=0)
    sensor_data = (sensor_data-mean)/var
    return sensor_data,label


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

