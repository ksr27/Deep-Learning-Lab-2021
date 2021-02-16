## Lydia ##

import tensorflow as tf
import gin
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io

@gin.configurable
def plot_confusion_matrix(cm, ds_info):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       ds_info : dict containing information about the dataset
    """
    figure = plt.figure(figsize=(8, 8))

    # Normalize the confusion matrix.
    cm = np.around(tf.cast(cm, tf.float32)/ tf.math.reduce_sum(cm, axis=0), decimals=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks_x = np.arange(cm.shape[1])
    tick_marks_y = np.arange(cm.shape[0])
    if ds_info['missing_classes']:
        class_names_x = np.array(ds_info['contained_classes']).astype(str)
    else:
        class_names_x = np.arange(start=1, stop=int(ds_info['num_classes'])).astype(str) # ignore 0 labels
    class_names_y = np.arange(start=1, stop=int(ds_info['num_classes'])).astype(str)  # ignore 0 labels
    plt.xticks(tick_marks_x, class_names_x, rotation=45)
    plt.yticks(tick_marks_y, class_names_y)

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
    Converts the matplotlib plot 'figure' to a PNG image for tensorboard log and storing to file
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer to a TF image. Make sure you use 4 channels for tensorboard.
    tb_image = tf.image.decode_png(buf.getvalue(), channels=4)
    # add batch dimension
    tb_image = tf.expand_dims(tb_image, 0)

    return tb_image, tf.image.decode_png(buf.getvalue(), channels=3)
