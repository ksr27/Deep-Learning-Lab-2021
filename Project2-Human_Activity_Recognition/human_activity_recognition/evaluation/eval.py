import datetime
import logging

import cv2
import gin
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluation.metrics import ConfusionMatrix, Accuracy, BalancedAccuracy, Precision, Recall, F1Score
from input_pipeline.visualize import plot_to_image, plot_confusion_matrix


@gin.configurable
class Evaluator(object):
    def __init__(self, model, ds_test, ds_info, visualize_flag, checkpoint, num_batches):
        """Evaluates model using test dataset.

        Parameters:
            model (keras.Model): keras model object
            checkpoint (string): checkpoint to load trained model params from
            ds_test (tf.data.Dataset): datasets with (image,label) pairs to run though trained model
            visualize_flag (bool): Flag to enable/disable deep visualization
            num_batches (int): amount of batches to use for visualization

        Returns:
            Nothing, evaluation results are logged to tensorboard and console
        """
        # restore from checkpoint
        self.model = model
        ckpt = tf.train.Checkpoint(net=self.model)
        status = ckpt.restore(checkpoint).expect_partial()

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.ds_test = ds_test
        self.ds_info = ds_info
        self.visualize_flag = visualize_flag
        self.num_batches = num_batches

        # init loss and metrics
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss = tf.keras.metrics.Mean()
        self.baccuracy = BalancedAccuracy(ds_info=ds_info)
        self.accuracy = Accuracy(ds_info=ds_info)
        self.confusion_matrix = ConfusionMatrix(ds_info=ds_info)
        self.recall = Recall(ds_info=ds_info)
        self.precision = Precision(ds_info=ds_info)
        self.f1_score = F1Score(ds_info=ds_info)

        # create summary writer for tensorboard logging
        self.summary_writer = tf.summary.create_file_writer("logs/eval/" + self.timestamp)

    @tf.function
    def eval_step(self, sensor_data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(sensor_data, training=False)
        eval_loss = self.loss_object(labels, predictions)
        self.loss.update_state(eval_loss)
        self.baccuracy.update_state(labels, predictions)
        self.accuracy.update_state(labels, predictions)
        self.confusion_matrix.update_state(labels, predictions)
        self.recall.update_state(labels, predictions)
        self.precision.update_state(labels, predictions)
        self.f1_score.update_state(labels, predictions)

    def visualize(self, sensor_data, label, prediction, iteration):
        """
        Plots the models prediction in comparison to the true label for given sensor data and stores the result to file
        """
        len_unbatched = self.ds_info['window_length'] * self.ds_info['batch_size']
        sensor_data = tf.reshape(sensor_data, shape=[len_unbatched, self.ds_info['num_features']])

        if len(tf.shape(label)) == 1:  # s2l
            prediction = tf.math.argmax(prediction, axis=1, output_type=tf.dtypes.int32)
            prediction = tf.reshape(tf.repeat(prediction, repeats=self.ds_info['window_length']),
                                    shape=[len_unbatched, ])
            label = tf.reshape(tf.repeat(label, repeats=self.ds_info['window_length']), shape=[len_unbatched, ])
        else:  # s2s
            prediction = tf.math.argmax(prediction, axis=2, output_type=tf.dtypes.int32)
            prediction = tf.reshape(prediction, shape=[len_unbatched, ])
            label = tf.reshape(label, shape=[len_unbatched, ])

        cmap = plt.cm.get_cmap('nipy_spectral', self.ds_info['num_classes'])

        norm = mpl.colors.BoundaryNorm(np.arange(self.ds_info['num_classes'] + 1), cmap.N)
        time_axis = tf.linspace(1, tf.shape(label)[0], tf.shape(label)[0])
        fig, axs = plt.subplots(5, 1, figsize=(8, 8))

        for i, sensor in enumerate(tf.transpose(sensor_data[:, :3])):
            axs[0].plot(time_axis.numpy(), sensor.numpy(), label=self.ds_info['sensor_mapping'][i])
            axs[0].grid(True)
            axs[1].plot(time_axis.numpy(), sensor.numpy(), label=self.ds_info['sensor_mapping'][i])
            axs[1].grid(True)

        axs[0].legend(loc='upper left', prop={'size': 6})
        axs[0].title.set_text('Accelerometer y_true')
        plot = axs[0].pcolorfast((0, len_unbatched), axs[0].get_ylim(), label.numpy()[np.newaxis], cmap=cmap, norm=norm,
                                 alpha=0.3)
        axs[1].legend(loc='upper left', prop={'size': 6})
        axs[1].title.set_text('Accelerometer y_pred')
        axs[1].pcolorfast((0, len_unbatched), axs[1].get_ylim(), prediction.numpy()[np.newaxis], cmap=cmap, norm=norm,
                          alpha=0.3)

        for i, sensor in enumerate(tf.transpose(sensor_data[:, 3:])):
            axs[2].plot(time_axis.numpy(), sensor.numpy(), label=self.ds_info['sensor_mapping'][i + 3])
            axs[2].grid(True)
            axs[3].plot(time_axis.numpy(), sensor.numpy(), label=self.ds_info['sensor_mapping'][i + 3])
            axs[3].grid(True)

        axs[2].legend(loc='upper left', prop={'size': 6})
        axs[2].title.set_text('Gyroscope y_true')
        axs[2].pcolorfast((0, len_unbatched), axs[2].get_ylim(), label.numpy()[np.newaxis], cmap=cmap, norm=norm,
                          alpha=0.3)
        axs[3].legend(loc='upper left', prop={'size': 6})
        axs[3].title.set_text('Gyroscope y_pred')
        axs[3].pcolorfast((0, len_unbatched), axs[3].get_ylim(), prediction.numpy()[np.newaxis], cmap=cmap, norm=norm,
                          alpha=0.3)

        cbar = fig.colorbar(plot, cax=axs[4], orientation="horizontal", aspect=20,
                            boundaries=np.arange(self.ds_info['num_classes'] + 1), values=np.arange(
                self.ds_info['num_classes']))
        cbar.set_ticks(np.arange(self.ds_info['num_classes']) + 0.5)
        cbar.ax.set_xticklabels(self.ds_info['label_mapping'], rotation=70)
        fig.tight_layout()
        __, image = plot_to_image(fig)
        cv2.imwrite("logs/eval/" + self.timestamp + '/visualization' + str(iteration.numpy()) + '.png',
                    cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR))

    def evaluate(self):
        # run model on sensor data and get loss+ metric values
        for (sensor_data, labels) in self.ds_test:
            self.eval_step(sensor_data, labels)

        # print loss and metric values to console
        template = 'Eval Loss: {}, Eval Balanced Accuracy: {},Eval Accuracy: {}, Eval Confusion Matrix: \n {}, ' \
                   '\n Eval Precision: {},\n Eval Recall {},\n Eval F1 Score: {}'
        logging.info(template.format(self.loss.result(),
                                     self.baccuracy.result() * 100,
                                     self.accuracy.result() * 100,
                                     self.confusion_matrix.result(),
                                     self.precision.result() * 100,
                                     self.recall.result() * 100,
                                     self.f1_score.result()))

        cm_image_tb, cm_image = plot_to_image(
            plot_confusion_matrix(self.confusion_matrix.result(), ds_info=self.ds_info))
        # save the confusion matrix to file
        cv2.imwrite("logs/eval/" + self.timestamp + '/cm.png', cv2.cvtColor(cm_image.numpy(), cv2.COLOR_RGB2BGR))

        # Write evaluation summary to tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar('Eval loss', self.loss.result(), step=0)
            tf.summary.scalar('Eval balanced accuracy', self.baccuracy.result(), step=0)
            tf.summary.scalar('Eval accuracy', self.accuracy.result(), step=0)
            tf.summary.image('Eval confusion matrix', cm_image_tb, step=0)

        # visualize the model prediction
        if self.visualize_flag:
            for i, (sensor_data, label) in self.ds_test.take(self.num_batches).enumerate():
                prediction = self.model(sensor_data, training=False)
                self.visualize(sensor_data, label, prediction, iteration=i)

        return
