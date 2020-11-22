import gin
import tensorflow as tf

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


def augment(image, label):
    """Data augmentation"""

    return image, label


