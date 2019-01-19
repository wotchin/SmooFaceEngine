import cv2
import numpy as np
from keras import backend as K


def rebuild_model(model, input_layer="input", output_layer="feature"):
    # if model is not an instance of Model, then try to load_model it by filepath.
    if isinstance(model, str):
        from model.amsoftmax import load_model
        model = load_model(model)

    __input_layer = model.get_layer(name=input_layer)
    __output_layer = model.get_layer(name=output_layer)
    func = K.function([__input_layer.input],
                      [__output_layer.output])
    input_shape = __input_layer.input_shape[1:]
    output_shape = __output_layer.output_shape[1:]
    return func, input_shape, output_shape


def get_feature_function(model, **kwargs):
    feature_function, input_shape, _ = rebuild_model(model, **kwargs)

    def inner(img):
        if isinstance(img, str):
            img = cv2.imread(img)
            assert img is not None
        if not isinstance(img, np.ndarray):
            raise Exception("img must be 'numpy.ndarray' type.But input #1 argument type is "
                            + str(type(img)))
        # preprocess
        img = cv2.resize(img, input_shape[:-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 256
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        vector = feature_function([img])[0]
        vector = vector.flatten()
        return vector

    return inner
