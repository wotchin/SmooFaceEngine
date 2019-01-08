import os
import time
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import random
import scipy.ndimage as ndi

from utils.preprocess import preprocess_image


# for training model
class DataGenerator(object):
    def __init__(self,
                 path,
                 batch_size,
                 input_size,
                 dataset,
                 is_shuffle=True,
                 data_augmentation=0,
                 translation_factor=0.3,
                 zoom_range=None,
                 validation_split=0):

        if zoom_range is None:
            zoom_range = [0.75, 1.25]
        data = []
        # [(x0,y0),(x1,y1),(x2,y2)...]
        keys = []
        # [y0,y1,y2...]
        if dataset == "lfw":
            g = os.walk(path)
            for item in g:
                name = item[0]
                keys.append(name)
                photo_list = item[2]
                for photo in photo_list:
                    img = cv2.imread(name + "/" + photo)
                    data.append((img, name))
        elif dataset == "olivettifaces":
            olive = cv2.imread(path)
            if olive is None:
                raise Exception("can not open the olivettifaces dataset file")
            label = 0
            count = 1
            keys = list(range(0, 40))
            for row in range(20):
                for column in range(20):
                    img = olive[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
                    data.append((img, label))
                    if count % 10 == 0 and count != 0:
                        label += 1
                    count += 1
        else:
            raise Exception("can not recognize this dataset")

        if data_augmentation > 0:
            self.zoom_range = zoom_range
            self.translation_factor = translation_factor
            data_size = len(data)
            expand_size = int(data_augmentation * data_size)
            expand_list = []
            for i in range(0, expand_size):
                index = i % data_size
                # 确保不会超过data_size的大小,然后在data List中循环遍历
                crop_img = self._do_random_crop(data[index][0])
                expand_list.append((crop_img, data[index][1]))
                rotate_img = self._do_random_rotation(data[index][0])
                expand_list.append((rotate_img, data[index][1]))
            data.extend(expand_list)
        if is_shuffle:
            random.shuffle(data)

        training_set_size = int(len(data) * (1 - validation_split))
        self.training_set = data[:training_set_size]
        self.validation_set = data[training_set_size:]
        self.keys = keys
        self.batch_size = batch_size
        self.f = open("./dataGenerator.log", "w+")
        self.input_size = input_size
        self.number = (len(keys), len(data), training_set_size, len(data) - training_set_size)

    def get_number(self):
        return self.number

    def __del__(self):
        self.f.close()

    def _write_line(self, line):
        self.f.write("[{0}] {1} \n".format(time.time(), line))
        self.f.flush()

    def flow(self, mode='train'):
        training_set = self.training_set
        validation_set = self.validation_set

        keys = self.keys
        self._write_line(str(keys))
        k = [i for i in range(0, len(keys))]
        one_hot = to_categorical(k, num_classes=len(keys))

        image_array = []
        targets = []

        while True:
            if mode == 'train':
                for x, y in training_set:
                    if x is None:
                        continue
                    img = preprocess_image(input_shape=self.input_size, image=x)
                    image_array.append(img)
                    targets.append(one_hot[keys.index(y)])

                    if len(targets) == self.batch_size:
                        image_array = np.asarray(image_array)
                        targets = np.asarray(targets)
                        yield self._wrap(image_array, targets)
                        image_array = []
                        targets = []

            elif mode == 'validate':
                for x, y in validation_set:
                    if x is None:
                        continue
                    img = preprocess_image(input_shape=self.input_size, image=x)
                    image_array.append(img)
                    targets.append(one_hot[keys.index(y)])

                    if len(targets) == self.batch_size:
                        image_array = np.asarray(image_array)
                        targets = np.asarray(targets)
                        yield self._wrap(image_array, targets)
                        image_array = []
                        targets = []

            else:
                raise Exception("unknown mode")
            # 如果还有剩余,即总数无法整除batch_size
            if len(targets) > 0:
                image_array = np.asarray(image_array)
                targets = np.asarray(targets)
                yield self._wrap(image_array, targets)
                image_array = []
                targets = []

    def _do_random_crop(self, image_array):
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                         self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                                                            crop_matrix, offset=offset, order=0, mode='nearest',
                                                            cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def _do_random_rotation(self, image_array):
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                         self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                                                            crop_matrix, offset=offset, order=0, mode='nearest',
                                                            cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def _wrap(self, image_array, targets):
        return [{'input': image_array},
                {'predictions': targets}]


def augment_data_to_dir():
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        cval=0,
        channel_shift_range=0,
        vertical_flip=False)

    def inner(img, prefix, dir_path):
        i = 0
        for _ in data_gen.flow(img, batch_size=1,
                               save_to_dir=dir_path,
                               save_prefix=prefix,
                               save_format='jpeg'):
            i += 1
            if i > 2:
                break  # 否则生成器会退出循环

    return inner
