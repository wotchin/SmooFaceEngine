from keras.layers import Activation, Convolution2D, Dropout, Conv2D,Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2


def smooNet_v1(input_shape,num_classes):
    #coded by wotchin
    #from VIPLFaceNet
    model = Sequential()
    # Conv layer 1 output shape (55, 55, 48)
    model.add(Conv2D(
        kernel_size=(9, 9), 
        activation="relu",
        filters=48, 
        strides=(4, 4), 
        input_shape=input_shape
    ))
    #pool1
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # Conv layer 2 output shape (27, 27, 128)
    model.add(Conv2D(
        strides=(1, 1), 
        kernel_size=(3,3), 
        activation="relu", 
        filters=128
    ))

    # Conv layer 3 output shape (13, 13, 192)
    model.add(Conv2D(
        strides=(1, 1), 
        kernel_size=(3,3), 
        activation="relu", 
        filters=128
    ))

    #pool2
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    #conv4
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=256,
        padding="same",
        strides=(1,1)
    ))

    #conv5
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=192,
        padding="same",
        strides=(1,1)
    ))

    #conv6
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=192,
        padding="same",
        strides=(1,1)
    ))

    #conv7
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=128,
        padding="same",
        strides=(1,1)
    ))

    #pool3
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
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # output
    model.add(Dense(num_classes))
    model.add(Activation('softmax',name='predictions'))
    #return 
    return model




def smooNet_v0(input_shape,num_classes):
    # from VGG
    model = Sequential()
    # Conv1,2
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=64, 
        strides=(1,1), 
        input_shape=input_shape
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=64, 
        strides=(1,1), 
    ))
    model.add(BatchNormalization())
    # pool1
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 3,4
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=128, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=128, 
        strides=(1,1), 
    ))
    model.add(BatchNormalization())
    # pool2
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 5-7
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))
    model.add(BatchNormalization())
    # pool3
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 8-10
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    model.add(BatchNormalization())
    # pool4
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 11-13
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #model.add(BatchNormalization())
    # pool5
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))

    # fully connected layer 1
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    # fully connected layer 2
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax',name='predictions'))
    #return 
    return model


def smooNet_v2(input_shape,num_classes):
    #coded by wotchin
    model = Sequential()
    model.add(Conv2D(
        kernel_size=(3,3), 
        activation="relu",
        filters=48, 
        strides=(1,1), 
        input_shape=input_shape
    ))
    model.add(Conv2D(
        kernel_size=(3,3), 
        activation="relu",
        filters=48, 
        strides=(1,1), 
    ))
    model.add(Conv2D(
        kernel_size=(3,3), 
        activation="relu",
        filters=48, 
        strides=(1,1), 
    ))
    model.add(Conv2D(
        kernel_size=(3,3), 
        activation="relu",
        filters=48, 
        strides=(1,1), 
    ))
    #pool1
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # Conv layer 2 output shape (27, 27, 128)
    model.add(Conv2D(
        strides=(1, 1), 
        kernel_size=(3,3), 
        activation="relu", 
        filters=128
    ))

    # Conv layer 3 output shape (13, 13, 192)
    model.add(Conv2D(
        strides=(1, 1), 
        kernel_size=(3,3), 
        activation="relu", 
        filters=128
    ))

    #pool2
    model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    #conv4
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=256,
        padding="same",
        strides=(1,1)
    ))

    #conv5
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=192,
        padding="same",
        strides=(1,1)))
    #conv6
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=192,
        padding="same",
        strides=(1,1)
    ))

    #conv7
    model.add(Conv2D(
        kernel_size=(3, 3),
        activation="relu", 
        filters=128,
        padding="same",
        strides=(1,1)
    ))

    #pool3
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
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # output
    model.add(Dense(num_classes))
    model.add(Activation('softmax',name='predictions'))
    #return 
    return model
