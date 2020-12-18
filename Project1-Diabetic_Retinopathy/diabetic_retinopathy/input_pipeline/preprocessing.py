import gin
import tensorflow as tf
#import imgaug.augmenters as iaa
#import tensorflow_addons as tfa
#import matplotlib.pyplot as plt


@tf.function
@gin.configurable
def preprocess(image, label, img_height, img_width,ds_name):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    image = tf.cast(image, tf.float32)/255.

    # Crop and pad image to square in central region (for idrid dataset)
    if ds_name == "idrid":
        image = tf.image.crop_to_bounding_box(image,offset_height=0,offset_width=270,target_height=2848,target_width=3440)
        image = tf.image.pad_to_bounding_box(image,offset_height=291,offset_width=0,target_height=3440,target_width=3440)

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    return image, label

@tf.function
def augment(image, label):
    """Data augmentation"""
    flipped_image = tf.image.flip_left_right(image)
    rotated_image = tf.image.rot90(flipped_image)
    #cropped_image = tf.image.central_crop(rotated_image, central_fraction=0.8)
    transposed_image= tf.image.transpose(rotated_image)
    #different_rotate = tfa.image.transform_ops.rotate(transposed_image, 1.0472)
    return transposed_image, label

