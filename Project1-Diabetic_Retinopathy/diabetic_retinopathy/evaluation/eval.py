import tensorflow as tf


def eval_step(model, images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    loss = model.loss_object(labels, predictions)
    model.eval_loss(loss)
    model.eval_accuracy(labels, predictions)

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    for (images,labels) in ds_test:
        eval_step(model,images,labels)

    template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(step,
                                 model.train_loss.result(),
                                 model.train_accuracy.result() * 100,
                                 model.test_loss.result(),
                                 model.test_accuracy.result() * 100))

    # Write summary to tensorboard
    with self.summary_writer.as_default():
        tf.summary.scalar('Train loss', self.train_loss.result(), step=step)
        tf.summary.scalar('Train accuracy', self.train_accuracy.result(), step=step)
        tf.summary.scalar('Test loss', self.test_loss.result(), step=step)
        tf.summary.scalar('Test accuracy', self.test_accuracy.result(), step=step)

    # Reset train metrics
    self.train_loss.reset_states()
    self.train_accuracy.reset_states()

    return