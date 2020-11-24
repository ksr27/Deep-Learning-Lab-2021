import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name="confusion_matrix"):
        super(ConfusionMatrix, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x =  [self.true_positives,self.false_positives,self.false_negatives,self.true_negatives]
        return tf.reshape(x,[2,2])

class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, name="accuracy"):
        super(Accuracy, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x =  (self.true_positives+self.true_negatives)/(self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)
        return x

class Sensitivity(tf.keras.metrics.Metric):

    def __init__(self, name="sensitivity"):
        super(Sensitivity, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...
       # y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x = self.true_positives/(self.true_positives+self.false_negatives)
        return x

class Specificity(tf.keras.metrics.Metric):

    def __init__(self, name="specificity"):
        super(Specificity, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...
       # y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x = self.true_negatives/(self.true_negatives+self.false_positives)
        return x

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score"):
        super(F1_Score, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...
       # y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x = 2*self.true_positives/(2*self.true_positives+self.false_positives+self.false_negatives)
        return x

class ROC_AUC(tf.keras.metrics.Metric):

    def __init__(self, name="roc_auc_score"):
        super(ROC_AUC, self).__init__(name=name)
        # ...
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred):
        # ...
       # y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))

        y_true = tf.cast(y_true, 'int32')
        y_pred =tf.cast(y_pred, 'int32')

        tp_values = tf.cast(tf.math.logical_and(( y_true==1), (y_pred==1)),'float32')
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        fp_values =  tf.cast(tf.math.logical_and((y_true==0) , (y_pred==1)),'float32')
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        fn_values =  tf.cast(tf.math.logical_and((y_true==1), (y_pred==0)),'float32')
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))
        tn_values =tf.cast( tf.math.logical_and((y_true==0) , (y_pred==0)),'float32')
        self.true_negatives.assign_add(tf.reduce_sum(tn_values))


    def result(self):
        # ...
        x = (self.true_positives)/(2*(self.true_positives+self.false_negatives))+(self.true_negatives)/(2*(self.false_positives+self.true_negatives))
        return x


# from metrics import ConfusionMatrix, Accuracy
