#Lydia

import tensorflow as tf
import datetime
import gin
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import cv2


@gin.configurable
def visualize(ds, num_pics):
    """
    Saves num_pics pictures from ds to file and tensorboard
    """
    logdir = "logs/img" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Sets up a timestamped log directory.
    file_writer = tf.summary.create_file_writer(logdir)  # Creates a file writer for the log directory.

    images = []
    for image, label in ds.take(num_pics):  # take num_pics random elements of ds
        cv2.imwrite("img" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png",
                    cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        images.append(image)

    # Using the file writer, log the images to tensorboard
    with file_writer.as_default():
        tf.summary.image("random image", images, max_outputs=num_pics, step=0)


@gin.configurable
def plot_cm(cm, class_names):
    """
    Returns a matplotlib figure with the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(tf.cast(cm, tf.float32) / tf.math.reduce_sum(cm, axis=0)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.tight_layout()
    return figure


@gin.configurable
def plot_to_image(figure):
    """
    Converts the matplotlib figure to a PNG image
    """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # convert image to png
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
