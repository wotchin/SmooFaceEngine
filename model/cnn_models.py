from keras import layers
from keras.layers import Activation, Dropout, Conv2D, Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import InputLayer, Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2


def VIPL_FaceNet(input_shape, num_classes):
    # from VIPLFaceNet
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape,
                         name="input"))
    # Conv layer 1 output shape (55, 55, 48)
    model.add(Conv2D(
        kernel_size=(9, 9),
        activation="relu",
        filters=48,
        strides=(4, 4)
    ))
    # pool1
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # Conv layer 2 output shape (27, 27, 128)
    model.add(Conv2D(
        strides=(1, 1),
        kernel_size=(3, 3),
        activation="relu",
        filters=128
    ))

    # Conv layer 3 output shape (13, 13, 192)
    model.add(Conv2D(
        strides=(1, 1),
        kernel_size=(3, 3),
        activation="relu",
        filters=128
    ))

    # pool2
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # conv4
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        padding="same",
        strides=(1, 1)
    ))

    # conv5
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=192,
        padding="same",
        strides=(1, 1)
    ))

    # conv6
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=192,
        padding="same",
        strides=(1, 1)
    ))

    # conv7
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=128,
        padding="same",
        strides=(1, 1)
    ))

    # pool3
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected layer 2
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation('relu', name="feature"))
    model.add(Dropout(0.5))

    # output
    model.add(Dense(num_classes))
    model.add(Activation('softmax', name='predictions'))

    # return
    return model


def VGG_Face(input_shape, num_classes):
    # from VGG
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape,
                         name="input"))

    # Conv1,2
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=64,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=64,
        strides=(1, 1),
    ))

    # pool1
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 3,4
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=128,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=128,
        strides=(1, 1),
    ))

    # pool2
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 5-7
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1),
    ))

    # pool3
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 8-10
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))
    # pool4
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 11-13
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1),
    ))

    # pool5
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # fully connected layer 2
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation('relu', name="feature"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax', name='predictions'))

    return model


# revised from face_classification
# we can see source cnn via:
# https://github.com/oarriaga/face_classification/blob/master/src/models/cnn.py
def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape, name="input")
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(8, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(1024, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, name="feature")(x)
    x = Dropout(.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output)

    return model


def ResNet(input_shape, num_classes):
    # wrap ResNet50 from keras
    from keras.applications.resnet50 import ResNet50
    input_tensor = Input(shape=input_shape, name="input")
    x = ResNet50(include_top=False,
                 weights=None,
                 input_tensor=input_tensor,
                 input_shape=None,
                 pooling="avg",
                 classes=num_classes)
    x = Dense(units=2048, name="feature")(x.output)
    return Model(inputs=input_tensor, outputs=x)
