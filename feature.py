import cv2
import keras
import numpy as np
from keras import backend as K
from numpy import linalg


class feature(object):
    def __init__(self, model_path,output_layer=20):
        model = keras.models.load_model(model_path)
        model.summary()
        self.feature_function = K.function([model.layers[0].input], [model.layers[output_layer].output])

    def get_vector(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,(64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        vector = self.feature_function([img])[0]
        return vector.flatten()

    def cosine_similarity(self, vector1, vector2):
        # val = vector1.dot(vector2.T)  # 若为列向量则 vector1.T * vector2
        val = np.dot(vector1, vector2)
        norm = linalg.norm(vector1) * linalg.norm(vector2)
        cos = val / norm  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return sim

    def euclidean_metric(self, vector1, vector2):
        dist = linalg.norm(vector1 - vector2)
        met = 1.0 / (1.0 + dist)  # 归一化
        return met

    def pearson_correlation(self, vector1, vector2):
        x_ = vector1 - np.mean(vector1)
        y_ = vector2 - np.mean(vector2)
        d = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
        return d
