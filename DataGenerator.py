import os
import time
from keras.utils import to_categorical
import cv2
import numpy as np


class DataGenerator(object):
    def __init__(self, path, batch_size, input_size, dataset):
        data = {}
        num_people = 0
        num_images = 0
        # {people:[pic1,pic2] ...}
        if dataset == "lfw":
            g = os.walk(path)
            for item in g:
                num_people += 1
                name = item[0]
                photo_list = item[2]
                bin_list = []
                for photo in photo_list:
                    img = cv2.imread(name + "/" + photo)
                    bin_list.append(img)
                    num_images += 1
                data[name] = bin_list
        elif dataset == "olivettifaces":
            olive = cv2.imread(path)
            if olive is None:
                raise Exception("can not open the olivettifaces file")
            label = 0
            count = 1
            num_images = 400
            num_people = 40
            bin_list = []
            for row in range(20):
                for column in range(20):
                    bin_list.append(olive[row*57:(row+1)*57,column*47:(column+1)*47])
                    if count % 10 == 0 and count != 0:
                        data[label] = bin_list
                        label += 1
                        bin_list = []
                    count += 1
        else:
            raise Exception("can not recognize this dataset")
        self.data = data
        self.batch_size = batch_size
        self.f = open("./dataGenerator.log", "w+")
        self.input_size = input_size
        self.number = (num_people, num_images)

    def get_number(self):
        return self.number

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
            for name in data:
                for photo in data[name]:
                    if photo is None:
                        continue
                    img = self._preprocess_image(image=photo)
                    image_array.append(img)
                    targets.append(one_hot[keys.index(name)])

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
        if self.input_size[-1] == 1:
            # 要求灰度图像
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        input_size = self.input_size[:2]
        image = cv2.resize(image, input_size)
        image = np.asarray(image, dtype='float64') / 256
        # 归一化
        if len(image.shape) < 3:
            image = np.expand_dims(image,-1)
        return image

    def _wrap(self, image_array, targets):
        return [{'conv2d_1_input': image_array},
                {'predictions': targets}]
