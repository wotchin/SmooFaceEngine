# https://cs.nyu.edu/~roweis/data.html
import cv2
import numpy as np
from utils.data_generator import augment_data_to_dir


def generate_more_faces(img_path, save_path):
    olive = cv2.imread(img_path)
    data_gen = augment_data_to_dir()
    if olive is None:
        raise Exception("can not open the olivettifaces dataset file.")
    for row in range(20):
        for column in range(2):
            img = olive[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
            img = np.expand_dims(img, 0)
            data_gen(img=img, prefix=str(row), dir_path=save_path)


def split_to_dataset(img_path, save_path):
    olive = cv2.imread(img_path)
    if olive is None:
        raise Exception("can not open the olivettifaces dataset file.")
    for row in range(20):
        for column in range(10):
            img = olive[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
            cv2.imwrite(img=img, filename=save_path + "/" + str(row) + "_" + str(column) + ".jpg")
