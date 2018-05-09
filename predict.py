import keras
from keras import backend as K
import numpy as np
from numpy import linalg
import cv2
import os


def get_feature_function(model_path, output_layer=20):
    model = keras.models.load_model(model_path)
    model.summary()
    vector_function = K.function([model.layers[0].input], [model.layers[output_layer].output])

    def inner(input_data):
        vector = vector_function([input_data])[0]
        return vector.flatten()

    return inner


def cosine_similarity(vector1, vector2):
    # val = vector1.dot(vector2.T)  # 若为列向量则 vector1.T * vector2
    val = np.dot(vector1, vector2)
    norm = linalg.norm(vector1) * linalg.norm(vector2)
    cos = val / norm  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def euclidean_metric(vector1, vector2):
    dist = linalg.norm(vector1 - vector2)
    met = 1.0 / (1.0 + dist)  # 归一化
    return met


def pearson_correlation(vector1, vector2):
    x_ = vector1 - np.mean(vector1)
    y_ = vector2 - np.mean(vector2)
    d = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
    return d


if __name__ == '__main__':
    get_feature = get_feature_function(model_path="./trained_models/smooFace.28-0.996528.hdf5", output_layer=20)
    features = []
    base_feature = None
    l = list(list(os.walk("./test"))[0])[2]
    l.sort()
    for file in l:
        path = "./test/" + file
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        feature = get_feature(img)
        features.append((file, feature))
        if file == "base.jpg":
            base_feature = feature

    for file, feature in features:
        print(file,
              cosine_similarity(feature, base_feature),
              euclidean_metric(feature, base_feature),
              pearson_correlation(feature, base_feature))
