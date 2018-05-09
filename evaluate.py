# /usr/bin env python

import cv2
from keras import preprocessing
from keras import backend as K
from keras.models import load_model
import numpy as np
import numpy.linalg as linalg
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import os

from keras.preprocessing.image import ImageDataGenerator


def get_data_augmentation():
    datagen = ImageDataGenerator(
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

    def inner(img, prefix):
        i = 0
        for batch in datagen.flow(img, batch_size=1,
                                  save_to_dir='olive', save_prefix=prefix, save_format='jpeg'):
            i += 1
            if i > 2:
                break  # 否则生成器会退出循环

    return inner


def gen_olive(path):
    olive = cv2.imread(path)
    datagen = get_data_augmentation()
    if olive is None:
        raise Exception("can not open the olivettifaces dataset file")
    keys = list(range(0, 40))
    data = []
    for row in range(20):
        for column in range(2):
            img = olive[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
            # img = np.expand_dims(img, 0)
            # datagen(img=img, prefix=str(row))
            cv2.imwrite(img=img, filename="./olive" + str(row) + "_" + str(column) + ".jpg")


def split_olive(path):
    olive = cv2.imread(path)
    if olive is None:
        raise Exception("can not open the olivettifaces dataset file")
    for row in range(20):
        for column in range(10):
            img = olive[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]
            cv2.imwrite(img=img, filename="./olive/split/" + str(row) + "_" + str(column) + ".jpg")


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    print(po, pe)
    return (po - pe) / (1 - pe)


def test_classifier():
    model = load_model("./trained_models/smooFace.28-0.996528.hdf5")
    files = list(os.walk("./olive"))[0][2]
    x_list = []
    y_list = []
    y_prediction = []
    total = 0
    correct = 0
    matrix = np.zeros(shape=(20, 20))
    for file in files:
        label = file.split("_")[0].replace("olive", "")
        img = cv2.imread("./olive/" + file)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 256
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        x_list.append(img)
        y = model.predict(x=img)
        y = int(np.argmax(y) / 2)
        # 近似
        y_correct = int(label)
        total += 1
        if y == y_correct:
            correct += 1
        matrix[y_correct][y] += 1

    k = kappa(matrix=matrix)
    print("total is {0} ,precise is {1},kappa is {2}".format(total, correct / total, k))


def cosine_similarity(vector1, vector2):
    # val = vector1.dot(vector2.T)  # 若为列向量则 vector1.T * vector2
    val = np.dot(vector1, vector2)
    norm = linalg.norm(vector1) * linalg.norm(vector2)
    cos = val / norm  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


def test_recognition():
    threshold = 0.83

    def preprocess(pic):
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        pic = cv2.resize(pic, (64, 64))
        pic = pic / 256
        pic = np.expand_dims(pic, -1)
        pic = np.expand_dims(pic, 0)
        return pic

    model = load_model("./trained_models/smooFace.28-0.996528.hdf5")
    f = K.function([model.layers[0].input], [model.layers[20].output])
    base_img = preprocess(cv2.imread("./olive/split/0_0.jpg"))
    base_feature = f([base_img])[0].flatten()

    y_true = []
    for i in range(200):
        if i < 10:
            y_true.append(1)  # True
        else:
            y_true.append(0)  # False
    y_score = []
    for label in range(20):
        for photo in range(10):
            file = "./olive/split/" + str(label) + "_" + str(photo) + ".jpg"
            img = preprocess(cv2.imread(file))
            img_feature = f([img])[0].flatten()
            sim = cosine_similarity(base_feature, img_feature)
            print("label:{0} - {1} ,sim : {2}".format(label, photo, sim))
            if sim > threshold:
                y_score.append(1)  # True
            else:
                y_score.append(0)  # False
    correct = 0
    # hamming distance
    for i in range(200):
        if y_true[i] == y_score[i]:
            correct += 1

    print("pre is " + str(correct / 200))
    fpr, tpr, t = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # gen_olive("./data/olivettifaces.jpg")
    # split_olive("./data/olivettifaces.jpg")
    test_recognition()
