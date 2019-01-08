import tensorflow as tf
from keras import backend as K
from keras.layers import Dropout
from keras.engine.topology import Layer
from keras.models import Model


class AMSoftmax(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        super(AMSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=0)

        cosine = K.dot(inputs, self.kernel)
        return cosine

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = super(AMSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def amsoftmax_loss(y_true, y_pred, scale=30.0, margin=0.35):
    label = tf.reshape(tf.argmax(y_true, axis=-1), shape=(-1, 1))
    label = tf.cast(label, dtype=tf.int32)
    batch_range = tf.reshape(tf.range(tf.shape(y_pred)[0]), shape=(-1, 1))  # 0~batchsize-1
    indices_of_ground_truth = tf.concat([batch_range, tf.reshape(label, shape=(-1, 1))],
                                        axis=1)  # 2columns vector, 0~batchsize-1 and label
    ground_truth_score = tf.gather_nd(y_pred, indices_of_ground_truth)  # score of groundtruth

    m = tf.constant(margin, name='m')
    s = tf.constant(scale, name='s')

    added_margin = tf.cast(tf.greater(ground_truth_score, m),
                           dtype=tf.float32) * m  # if ground_truth_score>m, ground_truth_score-m
    added_margin = tf.reshape(added_margin, shape=(-1, 1))
    added_embedding_feature = tf.subtract(y_pred, y_true * added_margin) * s  # s(cos_theta_yi-m), s(cos_theta_j)

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=added_embedding_feature)
    loss = tf.reduce_mean(cross_ent)
    return loss


def wrap_cnn(model, feature_layer, input_shape, num_classes):
    cnn = model(input_shape, num_classes)
    assert isinstance(cnn, Model)
    x = cnn.get_layer(name=feature_layer).output
    x = Dropout(.5)(x)
    output_layer = AMSoftmax(num_classes, name="predictions")(x)
    return Model(inputs=cnn.input, outputs=output_layer)


def load_model(filepath):
    import keras.models
    model = keras.models.load_model(filepath=filepath,
                                    custom_objects={"AMSoftmax": AMSoftmax,
                                                    "amsoftmax_loss": amsoftmax_loss})
    return model
