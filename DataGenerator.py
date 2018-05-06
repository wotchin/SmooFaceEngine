import os
import time
from keras.utils import to_categorical
import cv2
import numpy as np


class DataGenerator(object):
    def __init__(self, dir_path, batch_size):
        g = os.walk(dir_path)
        lfw = {}
        for item in g:
            lfw[item[0]] = item[2]
        self.data = lfw
        self.batch_size = batch_size
        self.f = open("./dataGenerator.log", "a+")

    def __del__(self):
        self.f.close()

    def write_line(self, line):
        self.f.write("[{0}] {1} \n".format(time.time(), line))
        self.f.flush()

    def flow(self):
        data = self.data
        keys = list(data.keys())
        self.write_line(str(keys))
        k = [i for i in range(0, len(keys))]
        one_hot = to_categorical(k, num_classes=len(keys))

        image_array = []
        targets = []

        while True:
            for people in data:
                for photo in data[people]:
                    img = cv2.imread(people + "/" + photo)
                    image_array.append(img)
                    targets.append(one_hot[keys.index(people)])

                    if len(targets) == self.batch_size:
                        image_array = np.asarray(image_array)
                        targets = np.asarray(targets)
                        yield self._wrap(image_array, targets)
                        image_array = []
                        targets = []

            # 如果还有剩余,即总数无法整除batch_size
            if len(targets) > 0:
                image_array = np.asarray(image_array)
                targets = np.asarray(targets)
                yield self._wrap(image_array, targets)
                image_array = []
                targets = []

    def _data_augmentation(self):
        pass

    def _preprocess_image(self, image):
        '''
        TODO:
             gray and resize etc.
        '''

        return image

    def _wrap(self, image_array, targets):
        return [{'conv2d_1_input': image_array},
                {'predictions': targets}]
