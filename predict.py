import os

import cv2

from utils.feature import get_feature_function
from utils.measure import *

model_path = "./trained_models/tiny_XCEPTION.hdf5"


def main():
    get_feature = get_feature_function(model=model_path)
    features = []
    base_feature = None
    dir_list = list(list(os.walk("./data/manual_testing"))[0])[2]
    dir_list.sort()
    for file in dir_list:
        path = "./data/manual_testing/" + file
        img = cv2.imread(path)
        feature = get_feature(img)
        features.append((file, feature))
        if file == "base.jpg":
            base_feature = feature

    for file, feature in features:
        print(file, '\t',
              cosine_similarity(feature, base_feature), '\t',
              euclidean_metric(feature, base_feature), '\t',
              pearson_correlation(feature, base_feature))


if __name__ == '__main__':
    main()
