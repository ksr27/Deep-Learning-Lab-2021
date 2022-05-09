import tensorflow as tf
import gin
from evaluation.metrics import ConfusionMatrix, BalancedAccuracy, Sensitivity, Specificity, F1Score
import logging
import datetime
from input_pipeline.visualize import plot_cm, plot_to_image
from deep_visualization.grad_cam import grad_cam_wbp

@gin.configurable
def evaluate(model, ds_test, ds_info, checkpoint, visualize_flag):
    """Evaluates model using test dataset.

    Parameters:
        model (keras.Model): keras model object
        checkpoint (string): checkpoint to load trained model params from
        ds_test (tf.data.Dataset): datasets with (image,label) pairs to run though trained model
        ds_info: dictionary containing information about the dataset
        visualize_flag (bool): Flag to enable/disable deep visualization

    Returns:
        Nothing, evaluation results are logged to tensorboard and console
    """

    # restore checkpoint
    ckpt = tf.train.Checkpoint(net=model)
    status = ckpt.restore(checkpoint).expect_partial()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # init loss and metrics
    loss = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits = True, name='eval_loss')
    accuracy = BalancedAccuracy()
    confusion_matrix = ConfusionMatrix()
    sensitivity = Sensitivity()
    specificity = Specificity()
    f1_score = F1Score()

    # create summary writer for tensorboard logging
    summary_writer = tf.summary.create_file_writer("logs/eval/"+timestamp)

    # run model on each image and get loss+ metric values
    for (images, labels) in ds_test:
        predictions = model(images, training=False)
        loss.update_state(labels, predictions)
        accuracy.update_state(labels, predictions)
        confusion_matrix.update_state(labels, predictions)
        sensitivity.update_state(labels, predictions)
        specificity.update_state(labels, predictions)
        f1_score.update_state(labels, predictions)

    # print loss and metrix values to console
    template = 'Eval Confusion Matrix: {}, Eval Loss: {}, Eval Accuracy: {}, Eval Sensitivity: {}, ' \
               'Eval Specificity: {}, Eval F1 Score: {}'
    logging.info(template.format(confusion_matrix.result(),
                                 loss.result(),
                                 accuracy.result() * 100,
                                 sensitivity.result()*100,
                                 specificity.result()*100,
                                 f1_score.result()))

    # Write evaluation summary to tensorboard
    with summary_writer.as_default():
        tf.summary.scalar('Eval loss', loss.result(), step=0)
        tf.summary.scalar('Eval accuracy', accuracy.result(), step=0)
        tf.summary.image('Eval confusion matrix',
                          plot_to_image(plot_cm(confusion_matrix.result(),class_names=['0', '1'])),step=0)
        tf.summary.scalar('Eval sensitivity', sensitivity.result(), step=0)
        tf.summary.scalar('Eval specificity', specificity.result(), step=0)
        tf.summary.scalar('Eval F1 Score', f1_score.result(), step=0)

    # run grad cam
    if visualize_flag:
        grad_cam_wbp(model, "conv2d_7", ds_test, ds_info, timestamp, 1)
        grad_cam_wbp(model, "conv2d_7", ds_test, ds_info, timestamp, 0)

    return
