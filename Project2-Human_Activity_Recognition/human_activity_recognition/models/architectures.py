import gin
import tensorflow as tf


@gin.configurable
def lstm_arch(input_shape, n_classes, mode, lstm_layers, lstm_units, dense_layers, dense_units, dropout_rate,
              attention):
    """Defines a RNN architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        mode (string): 's2s' or 's2l', defines whether to use S2S or S2L classification
        lstm_layers (int): number of LSTM layers
        lstm_units: number of units per LSTM layer
        dense_layers: number of dense layers
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate
        attention (boolean): attention flag to add temporal attention to the model (only for S2L)

    Returns:
        (keras.Model): keras model object
    """

    inputs = tf.keras.Input(input_shape)
    out = inputs
    if lstm_layers > 1:
        out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(out)

    if mode == 's2l':
        if attention:  # temporal attention
            out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(out)
            attention = tf.keras.layers.Dense(1, activation='tanh')(
                out)  # one neuron layer, weight and bias shapes calculated automatically
            attention = tf.keras.layers.Flatten()(attention)
            attention = tf.keras.layers.Activation('softmax')(attention)
            attention = tf.keras.layers.RepeatVector(lstm_units)(attention)  # (None, 250) to (None, lstm_units,250)
            attention = tf.keras.layers.Permute([2, 1])(
                attention)  # change shape from (None, lstm_units,250) to (None, 250, lstm_units)
            attention_out = tf.keras.layers.Multiply()([out, attention])  # multiply weight with each dimension
            out = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=-2), output_shape=(lstm_units,))(
                attention_out)
        else:
            out = tf.keras.layers.LSTM(lstm_units)(out)
    elif mode == 's2s':
        out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(out)

    out = tf.keras.layers.Dropout(dropout_rate)(out)

    if mode == 's2l':
        for i in range(dense_layers):
            out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
        outputs = tf.keras.layers.Dense(n_classes)(out)

    elif mode == 's2s':
        for i in range(dense_layers):
            dense_layer1 = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)
            out = tf.keras.layers.TimeDistributed(dense_layer1)(out)
        out = tf.keras.layers.TimeDistributed(dense_layer1)(out)
        dense_layer2 = tf.keras.layers.Dense(n_classes)
        out = tf.keras.layers.TimeDistributed(dense_layer2)(out)
        outputs = tf.keras.layers.Activation('linear')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='lstm_arch')
