import cv2
import numpy as np


def preprocess_image(input_shape, image):
    assert len(input_shape) == 3

    if input_shape[-1] == 1:
        # 要求灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_shape = input_shape[:2]
    image = cv2.resize(image, input_shape)
    image = np.asarray(image, dtype='float64') / 256
    # 归一化
    if len(image.shape) < 3:
        image = np.expand_dims(image, -1)
    return image
