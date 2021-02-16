import gin
import tensorflow as tf
import logging
import datetime
from evaluation.metrics import ConfusionMatrix, BalancedAccuracy, Accuracy, Precision, Recall, F1Score
from input_pipeline.visualize import plot_confusion_matrix, plot_to_image
from shutil import copyfile
import focal_loss as focal_loss
import cv2
import os


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_test, ds_info, epochs, gamma, beta, switch, log_cm):
        # Summary Writer
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer("logs/train/" + self.timestamp)

        # Loss configurations
        # loss weighting, simplifies to no weighting for beta = 0.0
        class_distribution = ds_info['train']['class_distribution']
        self.weights = [(1 - beta) / (1 - pow(beta, x)) for x in class_distribution]
        # alaways map loss of '0'-labeled samples to zero
        self.weights[0] = 0.0
        self.scce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                       reduction=tf.keras.losses.Reduction.NONE)
        self.focal_loss = focal_loss.SparseCategoricalFocalLoss(gamma=gamma, from_logits=True,
                                                                class_weight=self.weights,
                                                                reduction=tf.keras.losses.Reduction.NONE)
        # epoch to switch from scce loss to focal loss
        self.switch = switch
        # for validation and test loss
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model, optimizer=self.optimizer,iterator=iter(ds_train))
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts/' + self.timestamp, max_to_keep=epochs)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = Accuracy(ds_info=ds_info)
        self.train_baccuracy = BalancedAccuracy(ds_info=ds_info)
        self.train_cm = ConfusionMatrix(ds_info=ds_info)
        self.train_precision = Precision(ds_info=ds_info)
        self.train_recall = Recall(ds_info=ds_info)
        self.train_f1_score = F1Score(ds_info=ds_info)

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_baccuracy = BalancedAccuracy(ds_info=ds_info)
        self.val_accuracy = Accuracy(ds_info=ds_info)
        self.val_cm = ConfusionMatrix(ds_info=ds_info)
        self.val_precision = Precision(ds_info=ds_info)
        self.val_recall = Recall(ds_info=ds_info)
        self.val_f1_score = F1Score(ds_info=ds_info)

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.epochs = epochs
        self.current_epoch = 1
        self.log_cm = log_cm

    @tf.function
    def train_step(self, sensor_data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(sensor_data, training=True)
            # switching between scce and focal loss
            if self.current_epoch <= self.switch:
                loss = self.scce_loss(labels, predictions)
            else:
                loss = self.focal_loss(labels, predictions)

            # loss weighting
            for i in range(self.ds_info['num_classes']):
                loss = tf.where(labels == i, self.weights[i] * loss, loss)
            # in case batch contains only zero labeled samples avoid /0
            if tf.math.count_nonzero(labels) == 0:
                loss = 0.0
            else:
                loss = tf.reduce_sum(loss) / tf.cast(tf.math.count_nonzero(labels), dtype=loss.dtype)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_baccuracy.update_state(labels, predictions)
        self.train_accuracy.update_state(labels, predictions)
        self.train_cm.update_state(labels, predictions)
        self.train_precision.update_state(labels, predictions)
        self.train_recall.update_state(labels, predictions)
        self.train_f1_score.update_state(labels, predictions)

    @tf.function
    def val_step(self, sensor_data, labels):
        predictions = self.model(sensor_data, training=False)
        loss = self.loss_object(labels, predictions)

        self.val_loss.update_state(loss)
        self.val_baccuracy.update_state(labels, predictions)
        self.val_accuracy.update_state(labels, predictions)
        self.val_cm.update_state(labels, predictions)
        self.val_precision.update_state(labels, predictions)
        self.val_recall.update_state(labels, predictions)
        self.val_f1_score.update_state(labels, predictions)

    def train(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            logging.info("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")

        if self.ds_info['mode'] == 's2l':
            steps_per_epoch = tf.math.ceil(self.ds_info['train']['len'] / (self.ds_info['batch_size']))
        else:
            steps_per_epoch = tf.math.ceil(
                self.ds_info['train']['len'] / (self.ds_info['batch_size'] * self.ds_info['window_length']))

        for idx, (sensor_data, labels) in enumerate(self.ds_train):
            step = idx + 1
            self.train_step(sensor_data, labels)

            if step % steps_per_epoch == 0:

                # Reset val metrics
                self.val_loss.reset_states()
                self.val_baccuracy.reset_states()
                self.val_accuracy.reset_states()
                self.val_cm.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                # ROC AUC: {},, Val ROC AUC: {}
                template = 'Epoch {}/{}, Loss: {}, Balanced Accuracy: {}, Accuracy: {},\n Confusion Matrix: \n {}, ' \
                           '\n Precision: {} \n,\n Recall: {} \n, \n F1 Score: {}\n Val Loss: {}, Val Balanced Accuracy: {},' \
                           ' Val Accuracy: {},\n Val Confusion Matrix: \n {},\n Val Precision: {} \n, \n Val Recall: {} \n,' \
                           ' \n Val F1 Score: {}'
                logging.info(template.format(
                    self.current_epoch,
                    self.epochs,
                    self.train_loss.result(),
                    self.train_baccuracy.result() * 100,
                    self.train_accuracy.result() * 100,
                    self.train_cm.result(),
                    self.train_precision.result()*100,
                    self.train_recall.result() * 100,
                    self.train_f1_score.result(),

                    self.val_loss.result(),
                    self.val_baccuracy.result() * 100,
                    self.val_accuracy.result() * 100,
                    self.val_cm.result(),
                    self.val_precision.result() * 100,
                    self.val_recall.result() * 100,
                    self.val_f1_score.result()))

                # save cm imgs to file
                train_cm_tb, train_cm = plot_to_image(
                    plot_confusion_matrix(self.train_cm.result(), ds_info=self.ds_info))
                val_cm_tb, val_cm = plot_to_image(plot_confusion_matrix(self.val_cm.result(), ds_info=self.ds_info))

                if self.log_cm:
                    if self.current_epoch == 1:
                        os.mkdir("./logs/train/" + self.timestamp + "/train_cm")
                        os.mkdir("./logs/train/" + self.timestamp + "/val_cm")

                    cv2.imwrite("./logs/train/" + self.timestamp + '/train_cm/' + str(self.current_epoch) + '.png',
                                cv2.cvtColor(train_cm.numpy(), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("./logs/train/" + self.timestamp + '/val_cm/' + str(self.current_epoch) + '.png',
                                cv2.cvtColor(val_cm.numpy(), cv2.COLOR_RGB2BGR))

                # Write summary to tensorboard
                with self.summary_writer.as_default():
                    tf.summary.scalar('Train loss', self.train_loss.result(), step=self.current_epoch)
                    tf.summary.scalar('Train balanced accuracy', self.train_baccuracy.result(), step=self.current_epoch)
                    tf.summary.scalar('Train accuracy', self.train_accuracy.result(), step=self.current_epoch)
                    tf.summary.image('Train Confusion Matrix', train_cm_tb, step=self.current_epoch)

                    tf.summary.scalar('Val loss', self.val_loss.result(), step=self.current_epoch)
                    tf.summary.scalar('Val balanced accuracy', self.val_baccuracy.result(), step=self.current_epoch)
                    tf.summary.scalar('Val accuracy', self.val_accuracy.result(), step=self.current_epoch)
                    tf.summary.image('Val Confusion Matrix', val_cm_tb, step=self.current_epoch)

                # Reset train metrics
                self.train_loss.reset_states()
                log_train_acc = self.train_baccuracy.result().numpy()
                self.train_baccuracy.reset_states()
                self.train_cm.reset_states()

                # Save checkpoint each epoch
                save_path = self.manager.save()

                self.current_epoch += 1
                yield self.val_baccuracy.result().numpy()

            if self.current_epoch == self.epochs:
                logging.info(f'Finished training after {step} steps and {self.epochs} epochs.')
                # Save final checkpoint
                save_path = self.manager.save()
                copyfile('./configs/config.gin', './logs/train/' + self.timestamp + '/config.gin')

                return self.val_baccuracy.result().numpy()
