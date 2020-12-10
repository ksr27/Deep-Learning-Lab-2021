import cv2
import numpy as np
import tensorflow as tf
import datetime as datetime
import gin


@tf.custom_gradient
def guided_relu(x):
    y = tf.nn.relu(x)

    def grad(dy):
        pos_grads = tf.cast(dy > 0, "float32")  # only use positive derivatives (gradients)
        pos_filters = tf.cast(x > 0,"float32")  # only use positive conv filter values
        return pos_grads * pos_filters * dy

    return y, grad


@gin.configurable
def grad_cam_wbp(model, checkpoint, layer, ds, class_index, img_height, img_width, num_of_batches):
    # restore latest checkpoint
    ckpt = tf.train.Checkpoint(net=model)
    status = ckpt.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    # replace relu with custom gradient guided relu
    for layer in grad_model.layers[1:]:
        if hasattr(layer, 'activation'):
            layer.activation = guided_relu

    # create summary writer for tensorboard logging
    path = "logs/gradcam/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(path)

    for images, label in ds.take(num_of_batches):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(images)
            class_prediction = predictions[:, class_index]

        guided_grads = tape.gradient(class_prediction, conv_outputs)  # guided backpropagation
        weights = tf.reduce_mean(guided_grads, axis=(1, 2)) # gap of each filter

        # get grad cam output for each image in batch
        for j, batch_conv_layer in enumerate(conv_outputs):
            cam = tf.ones(batch_conv_layer.shape[0:2], dtype=tf.float32)
            for i, w in enumerate(weights[j, :]):
                cam += w * batch_conv_layer[:, :, i]

            cam = tf.squeeze(tf.image.resize(tf.expand_dims(cam,2), size=(img_height, img_width)))
            cam = tf.maximum(cam, 0)  # replace all negative values in cam with 0
            heatmap = (cam - tf.math.reduce_min(cam)) / (tf.math.reduce_max(cam) - tf.math.reduce_min(cam))  #norm

            cam = cv2.applyColorMap(np.uint8(255 * np.array(heatmap)), cv2.COLORMAP_JET)
            output_image = cv2.addWeighted(cv2.cvtColor(np.array(tf.cast(images[j, :, :, :] * 255, tf.uint8)),
                                                        cv2.COLOR_RGB2BGR), 0.7, cam, 0.5, 0)
            comp_image = cv2.cvtColor(np.array(tf.cast(images[j, :, :, :] * 255, tf.uint8)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(path + '/comp' + str(j) + '.png', comp_image)
            cv2.imwrite(path + '/cam_' + str(j) + '_class' + str(class_index) + '_label' + str(np.array(label))
                        + '_pred' + str(np.array(class_prediction))+ '.png', output_image)

            with summary_writer.as_default():
                tf.summary.image(
                    'Comparison' + str(j) + ' for class' + str(class_index) + ', label: ' + str(np.array(label)),
                    tf.expand_dims(cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB), axis=0),
                    step=0)
                tf.summary.image(
                    'GradCam' + str(j) + ' for class' + str(class_index) + ', label: ' + str(np.array(label)),
                    tf.expand_dims(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), axis=0),
                    step=0)


@gin.configurable
def grad_cam2(model, checkpoint, layer, ds, class_index, img_height, img_width, num_of_pics):
    # restore latest checkpoint
    ckpt = tf.train.Checkpoint(net=model)
    status = ckpt.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    # create summary writer for tensorboard logging
    path = "logs/gradcam/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(path)

    for images, label in ds.take(1):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(images)
            class_prediction = predictions[:, class_index]

        grads = tape.gradient(class_prediction, conv_outputs)  # guided_
        gate_f = tf.cast(conv_outputs > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_r * gate_f * grads

        weights = tf.reduce_mean(guided_grads, axis=(1, 2))

        for j, batch_conv_layer in enumerate(conv_outputs):
            cam = np.ones(batch_conv_layer.shape[0:2], dtype=np.float32)
            for i, w in enumerate(weights[j, :]):
                cam += w * batch_conv_layer[:, :, i]

            cam = cv2.resize(cam.numpy(), (img_height, img_width))
            cam = np.maximum(cam, 0)  # replaces all negative values in cam with 0
            heatmap = (cam - cam.min()) / (
                        cam.max() - cam.min())  # normalization of cam by removing min and div by max-min

            cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            cv2.imwrite(path + '/cam.png', cam)
            output_image = cv2.addWeighted(
                cv2.cvtColor(np.array(tf.cast(images[j, :, :, :] * 255, tf.uint8)), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
            comp_image = cv2.cvtColor(np.array(tf.cast(images[j, :, :, :] * 255, tf.uint8)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                path + '/comp' + str(j) + '_class' + str(class_index) + '_label' + str(np.array(label)) + '.png',
                comp_image)
            cv2.imwrite(
                path + '/cam_' + str(j) + '_class' + str(class_index) + '_label' + str(np.array(label)) + '.png',
                output_image)

            with summary_writer.as_default():
                tf.summary.image(
                    'Comparison' + str(j) + ' for class' + str(class_index) + ', label: ' + str(np.array(label)),
                    tf.expand_dims(cv2.cvtColor(comp_image, cv2.COLOR_BGR2RGB), axis=0),
                    step=0)
                tf.summary.image(
                    'GradCam' + str(j) + ' for class' + str(class_index) + ', label: ' + str(np.array(label)),
                    tf.expand_dims(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), axis=0),
                    step=0)
