import tensorflow as tf
import datetime
import gin
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io

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
        tf.summary.image("random image", images, max_outputs=num_pics, step = 0)

@gin.configurable
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

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
    cm = np.around(tf.cast(cm,tf.float32)/ tf.math.reduce_sum(cm,axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

@gin.configurable
def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image
