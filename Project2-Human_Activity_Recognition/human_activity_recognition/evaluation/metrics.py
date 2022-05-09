import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        return self.confusion_matrix

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))


class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        x = tf.reduce_sum(tf.linalg.diag_part(self.confusion_matrix)) / tf.reduce_sum(self.confusion_matrix)
        return x

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))


class BalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], dense_shape=shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        x = 0
        for j in range(len(self.contained_classes)):  # col: true classes
            x += self.confusion_matrix[j, j] / tf.reduce_sum(self.confusion_matrix[:, j])
        return x / self.num_classes

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))


class Precision(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="precision", **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        x = tf.linalg.diag_part(self.confusion_matrix) / tf.reduce_sum(self.confusion_matrix, axis=0)
        return x

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))


class Recall(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="recall", **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        x = tf.linalg.diag_part(self.confusion_matrix) / tf.reduce_sum(self.confusion_matrix, axis=1)
        return x

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, ds_info, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        if ds_info['missing_classes']:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = ds_info['contained_classes']
        else:
            self.num_classes = ds_info['num_classes'] - 1
            self.contained_classes = tf.range(1, ds_info['num_classes'])
        self.confusion_matrix = self.add_weight(shape=(self.num_classes, len(self.contained_classes)), name='cm',
                                                initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'int32')
        y_pred = tf.nn.softmax(y_pred)
        if len(tf.shape(y_true)) == 1:
            y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
        else:  # s2s
            y_pred = tf.math.argmax(y_pred, axis=2, output_type=tf.dtypes.int32)
        shape = [self.num_classes, len(self.contained_classes)]

        for i in range(1, self.num_classes + 1):  # rows = predicted classes
            for j in self.contained_classes:  # cols = true classes
                values = tf.cast(tf.math.logical_and((y_pred == i), (y_true == j)), 'float32')
                values = tf.reduce_sum(values)
                indices = [[i - 1, j - 1]]
                delta = tf.sparse.SparseTensor(indices, [values], shape)
                delta = tf.sparse.to_dense(delta)
                self.confusion_matrix.assign_add(delta)

    def result(self):
        x = 2 * tf.linalg.diag_part(self.confusion_matrix) / (
                tf.reduce_sum(self.confusion_matrix, axis=1) + tf.reduce_sum(self.confusion_matrix, axis=0))
        return x

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros(shape=(self.num_classes, len(self.contained_classes))))
