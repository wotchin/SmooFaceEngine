# need dlib 19.9 or later
# you can download the file 'mmod_human_face_detector.dat' at http://dlib.net/files/


import dlib
import cv2

img = cv2.imread('lena.jpg')
hog_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
face_rects = hog_face_detector(img, 0)
for faceRect in face_rects:
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()
