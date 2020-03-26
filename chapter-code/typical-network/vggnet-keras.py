# this is a VGGNet demo.


def vgg_net(input_shape,num_classes):
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

    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=64, 
        strides=(1,1), 
    ))
    
    # pool1
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 3,4
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=128, 
        strides=(1,1), 
    ))
    #
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=128, 
        strides=(1,1), 
    ))
    
    # pool2
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 5-7
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))
    #
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))

    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=256, 
        strides=(1,1), 
    ))

    # pool3
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 8-10
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))
    #
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))

    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))

    # pool4
    model.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))


    # Conv 11-13
    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))

    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))

    model.add(Conv2D(
        kernel_size=(3, 3), 
        activation="relu",
        filters=512, 
        strides=(1,1), 
    ))

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
	
    return model
