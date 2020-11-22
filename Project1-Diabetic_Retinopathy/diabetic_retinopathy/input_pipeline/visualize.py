import tensorflow as tf
import datetime
import gin

@gin.configurable
def visualize(ds, img_height, img_width, num_pics):
    logdir = "logs/img" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Sets up a timestamped log directory.
    file_writer = tf.summary.create_file_writer(logdir)  # Creates a file writer for the log directory.

    images = []
    for image, label in ds.take(num_pics):  # take 3 random elements of ds
        image = tf.cast(image*255, tf.uint8) #scale back to 0-255 and convert to uint
        images.append(image)

    # Using the file writer, log the images to tensorboard
    with file_writer.as_default():
        tf.summary.image("random image", images, max_outputs=num_pics)