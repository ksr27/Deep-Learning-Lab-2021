import tensorflow as tf
import gin
from evaluation.metrics import ConfusionMatrix, Accuracy, Sensitivity, Specificity, F1Score, RocAuc
import logging
import datetime
from input_pipeline.visualize import plot_confusion_matrix, plot_to_image

@gin.configurable
def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer,net=model, iterator=iter(ds_test))
    ckpt.restore(checkpoint)

    # init loss and metrics
    loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='eval_loss', from_logits=True)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    confusion_matrix = ConfusionMatrix()
    sensitivity = Sensitivity()
    specificity = Specificity()
    f1_score = F1Score()
    roc_auc = RocAuc()

    # create summary writer for tensorboard logging
    summary_writer = tf.summary.create_file_writer("logs/eval/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

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
                                 sensitivity.result(),
                                 specificity.result(),
                                 f1_score.result(),
                                 roc_auc.result()))

    # Write evaluation summary to tensorboard
    with summary_writer.as_default():
        tf.summary.scalar('Eval loss', loss.result(), step=0)
        tf.summary.scalar('Eval accuracy', accuracy.result(), step=0)
        tf.summary.image('Eval confusion matrix',
                          plot_to_image(plot_confusion_matrix(confusion_matrix.result(),class_names=['1', '0'])),
                          step=0)
        tf.summary.scalar('Eval sensitivity', sensitivity.result(), step=0)
        tf.summary.scalar('Eval specificity', specificity.result(), step=0)
        tf.summary.scalar('Eval F1 Score', f1_score.result(), step=0)
        tf.summary.scalar('Eval ROC AUC', roc_auc.result(), step=0)
    return accuracy.result()
