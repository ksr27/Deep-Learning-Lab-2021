import tensorflow as tf
import gin
from evaluation.metrics import ConfusionMatrix, Accuracy, Sensitivity, Specificity, F1Score, RocAuc
import logging
import datetime

@gin.configurable
def evaluate(model, checkpoint, ds_test, visualize_flag):
    """Evaluates model using test dataset.

    Parameters:
        model (keras.Model): keras model object
        checkpoint (string): checkpoint to load trained model params from
        ds_test (tf.data.Dataset): datasets with (image,label) pairs to run though trained model
        visualize_flag (bool): Flag to enable/disable deep visualization

    Returns:
        Nothing, evaluation results are logged to tensorboard and console
    """
    # restore latest checkpoint
    ckpt = tf.train.Checkpoint(net=model)
    status = ckpt.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # init loss and metrics
    loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='eval_loss', from_logits=True)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    confusion_matrix = ConfusionMatrix()
    sensitivity = Sensitivity()
    specificity = Specificity()
    f1_score = F1Score()
    roc_auc = RocAuc()

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
        roc_auc.update_state(labels, predictions)

    # print loss and metrix values to console
    template = 'Eval Loss: {}, Eval Accuracy: {}, Eval Confusion Matrix: {}, Eval Sensitivity: {}, ' \
               'Eval Specificity: {}, Eval F1 Score: {}, Eval ROC AUC: {}'
    logging.info(template.format(loss.result(),
                                 accuracy.result() * 100,
                                 confusion_matrix.result(),
                                 sensitivity.result()*100,
                                 specificity.result()*100,
                                 f1_score.result()*100,
                                 roc_auc.result()))

    # Write evaluation summary to tensorboard
    with summary_writer.as_default():
        tf.summary.scalar('Eval loss', loss.result(), step=0)
        tf.summary.scalar('Eval accuracy', accuracy.result(), step=0)
        #tf.summary.image('Eval confusion matrix',
        #                  plot_to_image(plot_confusion_matrix(confusion_matrix.result(),class_names=['1', '0'])),
        #                  step=0)
        tf.summary.scalar('Eval sensitivity', sensitivity.result(), step=0)
        tf.summary.scalar('Eval specificity', specificity.result(), step=0)
        tf.summary.scalar('Eval F1 Score', f1_score.result(), step=0)
        tf.summary.scalar('Eval ROC AUC', roc_auc.result(), step=0)

    return
