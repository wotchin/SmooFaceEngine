from keras import layers
from keras.layers import Activation, Dropout, Conv2D, Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import InputLayer, Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import ZeroPadding2D, Add
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2


# from VIPLFaceNet
# You can see the paper at:
# https://arxiv.org/abs/1609.03892
def VIPL_FaceNet(input_shape, num_classes):
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


# implement VGGNet
def VGGNet(input_shape, num_classes):
    # Because VGGNet is more deep and there is many max pooling in the network,
    # I do not suggest you inputting too small value of input_shape.
    # The input shape of raw paper is (224, 224, 3)
    assert input_shape[0] >= 224 and input_shape[1] >= 224

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
        strides=(1, 1)))

    # pool1
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 3,4
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=128,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=128,
        strides=(1, 1)))

    # pool2
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))

    # Conv 5-7
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=256,
        strides=(1, 1)))

    # pool3
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))
    # Conv 8-10
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))

    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))
    # pool4
    model.add(MaxPooling2D((2, 2), strides=(2, 2),
                           padding='same'))
    # Conv 11-13
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu",
        filters=512,
        strides=(1, 1)))

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
# we can see source code at:
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


def ResNet50(input_shape, num_classes):
    # wrap ResNet50 from keras, because ResNet50 is so deep.
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


# implement ResNet's block.
# I implement two classes block:
# one is basic block, the other is bottleneck block.
def basic_block(filters, kernel_size=3, is_first_block=True):
    stride = 1
    if is_first_block:
        stride = 2

    def f(x):
        # f(x) named y
        # 1st Conv
        y = ZeroPadding2D(padding=1)(x)
        y = Conv2D(filters, kernel_size, strides=stride, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        # 2nd Conv
        y = ZeroPadding2D(padding=1)(y)
        y = Conv2D(filters, kernel_size, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)

        # f(x) + x
        if is_first_block:
            shortcut = Conv2D(filters, kernel_size=1, strides=stride, kernel_initializer='he_normal')(x)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = Add()([y, shortcut])
        y = Activation("relu")(y)

        return y

    return f


# ResNet v1, we can see the paper at:
# https://arxiv.org/abs/1512.03385
def ResNet18(input_shape, num_classes):
    input_layer = Input(shape=input_shape, name="input")

    # Conv1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv2
    x = basic_block(filters=64)(x)
    x = basic_block(filters=64, is_first_block=False)(x)

    # Conv3
    x = basic_block(filters=128)(x)
    x = basic_block(filters=128, is_first_block=False)(x)

    # Conv4
    x = basic_block(filters=256)(x)
    x = basic_block(filters=256, is_first_block=False)(x)

    # Conv5
    x = basic_block(filters=512)(x)
    x = basic_block(filters=512, is_first_block=False)(x)

    x = GlobalAveragePooling2D(name="feature")(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)
    return model
