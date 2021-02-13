import gin
import tensorflow as tf
import logging
import datetime
from evaluation.metrics import ConfusionMatrix, BalancedAccuracy, Sensitivity, Specificity, F1Score
from input_pipeline.visualize import plot_cm, plot_to_image
from shutil import copyfile


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_test, ds_info, run_paths, epochs,
                 log_interval, ckpt_interval, initial_lr, momentum):  # total_steps
        # Summary Writer
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer("logs/train/" + self.timestamp)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr, decay_steps=ds_info['size'], decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=momentum, nesterov=True)

        # Checkpoint Manager #Baran
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model, optimizer=self.optimizer,
                                        iterator=iter(ds_train))  #
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts/' + self.timestamp, max_to_keep=epochs)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = BalancedAccuracy()
        self.train_cm = ConfusionMatrix()
        self.train_sensitivity = Sensitivity()
        self.train_specificity = Specificity()
        self.train_f1_score = F1Score()

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = BalancedAccuracy()
        self.val_cm = ConfusionMatrix()
        self.val_sensitivity = Sensitivity()
        self.val_specificity = Specificity()
        self.val_f1_score = F1Score()

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.ds_train_size = ds_info['size']
        self.run_paths = run_paths
        self.epochs = epochs
        self.current_epoch = 1
        self.batch_size = ds_info['batch_size']
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_cm.update_state(labels, predictions)
        self.train_sensitivity.update_state(labels, predictions)
        self.train_specificity.update_state(labels, predictions)
        self.train_f1_score.update_state(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)
        self.val_cm.update_state(labels, predictions)
        self.val_sensitivity.update_state(labels, predictions)
        self.val_specificity.update_state(labels, predictions)
        self.val_f1_score.update_state(labels, predictions)

    def train(self, restore_ckpt=True):
        if restore_ckpt:
            self.ckpt.restore(self.manager.latest_checkpoint)

        if self.manager.latest_checkpoint:
            logging.info("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")

        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % tf.math.ceil(self.ds_train_size / self.batch_size) == 0:
                # Reset val metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()
                self.val_cm.reset_states()
                self.val_sensitivity.reset_states()
                self.val_specificity.reset_states()
                self.val_f1_score.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Epoch {}/{}, \n Confusion Matrix: {}, Loss: {:0.2f}, Accuracy: {:0.2f}, ' \
                           'Sensitivity: {:0.2f}, Specificity: {:0.2f}, F1 Score: {} \n Val Confusion Matrix: {}, ' \
                           'Val Loss: {:0.2f}, Val Accuracy: {:0.2f}\n {}, Val Sensitivity: {:0.2f}, ' \
                           'Val Specificity: {:0.2f}, Val F1 Score: {}'
                logging.info(template.format(
                    self.current_epoch,
                    self.epochs,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.train_cm.result(),
                    self.train_sensitivity.result() * 100,
                    self.train_specificity.result() * 100,
                    self.train_f1_score.result(),

                    self.val_loss.result(),
                    self.val_accuracy.result() * 100,
                    self.val_cm.result(),
                    self.val_sensitivity.result() * 100,
                    self.val_specificity.result() * 100,
                    self.train_f1_score.result()))

                # Write summary to tensorboard
                with self.summary_writer.as_default():
                    tf.summary.scalar('Train loss', self.train_loss.result(), step=self.current_epoch)
                    tf.summary.scalar('Train accuracy', self.train_accuracy.result(), step=self.current_epoch)
                    tf.summary.image('Train Confusion Matrix',
                                     plot_to_image(plot_cm(self.train_cm.result(), class_names=['0', '1'])),
                                     step=self.current_epoch)
                    tf.summary.scalar('Train sensitivity', self.train_sensitivity.result(), step=self.current_epoch)
                    tf.summary.scalar('Train specificity', self.train_specificity.result(), step=self.current_epoch)
                    tf.summary.scalar('Train F1 Score', self.train_f1_score.result(), step=self.current_epoch)

                    tf.summary.scalar('Val loss', self.val_loss.result(), step=self.current_epoch)
                    tf.summary.scalar('Val accuracy', self.val_accuracy.result(), step=self.current_epoch)
                    tf.summary.image('Validation Confusion Matrix',
                                     plot_to_image(plot_cm(self.val_cm.result(), class_names=['0', '1'])),
                                     step=self.current_epoch)
                    tf.summary.scalar('Val sensitivity', self.val_sensitivity.result(), step=self.current_epoch)
                    tf.summary.scalar('Val specificity', self.val_specificity.result(), step=self.current_epoch)
                    tf.summary.scalar('Val F1 Score', self.val_f1_score.result(), step=self.current_epoch)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.train_cm.reset_states()
                self.train_sensitivity.reset_states()
                self.train_specificity.reset_states()
                self.train_f1_score.reset_states()

                # Save checkpoint #Baran
                self.manager.save()

                self.current_epoch += 1
                yield self.val_accuracy.result().numpy()

            if self.current_epoch == self.epochs:
                logging.info(f'Finished training after {step} steps and {self.epochs} epochs.')
                # Save final checkpoint
                self.manager.save()
                copyfile('./configs/config.gin', './logs/train/' + self.timestamp + '/config.gin')
                return self.val_accuracy.result().numpy()
