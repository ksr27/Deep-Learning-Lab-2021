import gin
import tensorflow as tf
import numpy as np
import cv2
#import mclahe as mc

def histogram_equalization(img):
    img_in = np.array(img)
    hsv = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

@gin.configurable
@tf.function
def he_tf(img, img_height, img_width):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    r = he_channel(r)
    g = he_channel(g)
    b = he_channel(b)
    return tf.reshape([r,g,b],[img_height,img_width,3])

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
def preprocess1(image, label, img_height, img_width,ds_name):
    """Dataset preprocessing: Normalizing and resizing"""

    # Crop and pad image to square in central region (for idrid dataset)
    if ds_name == "idrid":
        image = tf.image.crop_to_bounding_box(image,offset_height=0,offset_width=270,target_height=2848,target_width=3440)
        image = tf.image.pad_to_bounding_box(image,offset_height=291,offset_width=0,target_height=3440,target_width=3440)

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    image = tf.cast(image, tf.uint8)
    # Normalize image: `uint8` -> `float32`.
    # image = tf.cast(image, tf.float32)/255.

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

    #flipped_image = tf.image.flip_left_right(image)
    #rotated_image = tf.image.rot90(flipped_image)
    #cropped_image = tf.image.central_crop(rotated_image, central_fraction=0.8)
    return image,label#cropped_image, label


