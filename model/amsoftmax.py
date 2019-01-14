import tensorflow as tf
from keras import backend as K
from keras.layers import Dropout
from keras.engine.topology import Layer
from keras.models import Model


class AMSoftmax(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.kernel = None
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        super(AMSoftmax, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # get cosine similarity
        # cosine = x * w / (||x|| * ||w||)
        inputs = K.l2_normalize(inputs, axis=1)
        kernel = K.l2_normalize(self.kernel, axis=0)
        cosine = K.dot(inputs, kernel)
        return cosine

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def get_config(self):
        config = {
            'units': self.units}
        base_config = super(AMSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# reference:
# https://github.com/hao-qiang/AM-Softmax/blob/master/AM-Softmax.ipynb
def amsoftmax_loss(y_true, y_pred, scale=30.0, margin=0.35):
    label = K.reshape(K.argmax(y_true, axis=-1), shape=(-1, 1))
    label = K.cast(label, dtype=tf.int32)

    batch_range = K.reshape(tf.range(K.shape(y_pred)[0]), shape=(-1, 1))
    # concat the two column vectors, one is the batch_range, the other is label.
    indices_of_ground_truth = tf.concat([batch_range, K.reshape(label, shape=(-1, 1))], axis=1)
    # score of ground truth
    ground_truth_score = tf.gather_nd(y_pred, indices_of_ground_truth)
    # make two const variables as tensor type.
    m = K.constant(margin, name='m')
    s = K.constant(scale, name='s')

    # if ground_truth_score > m, group_truth_score = group_truth_score - m
    added_margin = K.cast(K.greater(ground_truth_score, m), dtype=tf.float32) * m
    added_margin = K.reshape(added_margin, shape=(-1, 1))

    # s * (cos - m) if yi != j
    added_embedding_feature = tf.subtract(y_pred, y_true * added_margin) * s

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                               logits=added_embedding_feature)
    loss = tf.reduce_mean(cross_entropy)
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
