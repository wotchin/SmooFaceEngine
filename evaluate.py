# /usr/bin env python

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from data.olivetti_faces.split_img import split_to_dataset
from model.amsoftmax import load_model
from utils.feature import get_feature_function
from utils.measure import kappa, cosine_similarity

model_path = "./trained_models/tiny_XCEPTION.hdf5"
img_path = "./data/olivetti_faces/olivettifaces.jpg"
test_data_path = "./olive"
input_shape = (64, 64, 1)


def test_classifier():
    model = load_model(filepath=model_path)
    files = list(os.walk(test_data_path))[0][2]
    x_list = []
    total = 0
    correct = 0
    matrix = np.zeros(shape=(20, 20))
    for file in files:
        label = file.split("_")[0].replace("olive", "")
        img = cv2.imread(test_data_path + file)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 256
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)
        x_list.append(img)
        y = model.predict(x=img)
        y = int(np.argmax(y) / 2)

        y_correct = int(label)
        total += 1
        if y == y_correct:
            correct += 1
        matrix[y_correct][y] += 1

    k = kappa(matrix=matrix)
    print("total is {0}, precise is {1}, kappa is {2}."
          .format(total, correct / total, k))


def test_recognition():
    # This threshold is used to determine if two face images belong to the same person.
    threshold = 0.80

    model = load_model(filepath=model_path)
    f = get_feature_function(model)
    base_feature = f(cv2.imread("./olive/0_0.jpg"))

    y_true = []
    for i in range(200):
        if i < 10:
            y_true.append(1)  # True
        else:
            y_true.append(0)  # False
    y_score = []
    for label in range(20):
        for photo in range(10):
            file = "./olive/" + str(label) + "_" + str(photo) + ".jpg"
            img_feature = f(cv2.imread(file))
            sim = cosine_similarity(base_feature, img_feature)
            print("label:{0} - {1} ,sim : {2}".format(label, photo, sim))
            if sim > threshold:
                y_score.append(1)  # True
            else:
                y_score.append(0)  # False
    correct = 0
    for i in range(200):
        if y_true[i] == y_score[i]:
            correct += 1

    print("acc is " + str(correct / 200))
    fpr, tpr, t = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(test_data_path):
        os.mkdir(test_data_path)
    # generate_more_faces(, test_data_path)
    split_to_dataset(img_path, test_data_path)
    test_recognition()
