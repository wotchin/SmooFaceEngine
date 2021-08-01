# SmooFaceEngine
An open source face recognition engine.

Support for Tensorflow-2.0.0+ @[branch tensorflow2.0.0+]()

[Read more relational papers](https://github.com/wotchin/paper-list/blob/master/computer-vision.md)

[勘误表(errate)](https://github.com/wotchin/SmooFaceEngine/wiki/errata) 

[Chinese Wiki](https://github.com/wotchin/SmooFaceEngine/wiki)

# Introduction
Let us see something about this project now.
## What is this project?
This project is an open source project about
 face recognition. In the project, we implemente a face 
 recognition engine with one-shot training.

# Principle of SmooFace
In this project,
 we implemente some CNNs (VGGNet, VIPL face net, ResNet, XCEPTION, et al) to recognize face image.

Here, we use AM-Softmax loss as the cost function rather than 
triple loss or other metric learning loss functions because AM-Softmax has less training time but accuracy is still good.

# How to use?
This project is **only a demo**. In order to see the experimental results, we trained a model 
with small data. We use data augmentation in this project, so that we can get 
a robust model.
If you want to use this project in your production environment, you should **do more**.
## Train
```python3 train.py```
## Predict
```python3 predict.py```
## Web API
>http://127.0.0.1:8080/test

## Dependencies
    Python 3.6+
    Others: 
    ```
    pip3 install -r requirements.txt
    ```
    
# Reference
## Papers
You can search the following papers in [Google Scholar](https://scholar.google.com/)

    AM-Softmax
    Sphere face
    FaceNet
    ResNet
    Xception
    MobileNet v1,v2,v3
    VIPL Face net


## Open source projects

1. https://github.com/xiangrufan/keras-mtcnn
2. https://github.com/happynear/AMSoftmax
3. https://github.com/Joker316701882/Additive-Margin-Softmax
4. https://github.com/hao-qiang/AM-Softmax
5. https://github.com/ageitgey/face_recognition
6. https://github.com/oarriaga/face_classification
7. https://github.com/seetaface/SeetaFaceEngine
8. https://github.com/jiankangdeng/handbook

# LICENSE
Apache license version 2.0
# How to contribute
  There are many bugs here, so you could send some pull requests or give some issues for this project. Thank you very much :)
## TODO

1. give train.py arguments: for different training set
2. refactor: to optimize code
3. etc.
