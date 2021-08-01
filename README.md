# SmooFaceEngine
An open source face recognition engine named SmooFaceEngine.

[Read more relational papers](https://github.com/wotchin/paper-list/blob/master/computer-vision.md)

[勘误表(errate)](https://github.com/wotchin/SmooFaceEngine/wiki/errata) 

[Chinese Wiki](https://github.com/wotchin/SmooFaceEngine/wiki)
# Introduction
Let us see something about this project now.
## What is this project?
This project is an open source project about
 face recognition. In the project, we implemented a face 
 recognition engine that was one-shot training.

## Why did this?
JUST FOR FUN!
I think I may be not a face recognition expert, but a geeker.

# Principle of SmooFace
In this project,
 we implemented some CNN, such as VGGNet, VIPL face net, 
 ResNet and XCEPTION etc.

We used AM-Softmax loss as the cost function here, rather than 
triple loss or other metric learning loss functions. This is because 
AM-Softmax has less training time, but the accuracy is still good.

# How to use?
This project is a demo. In order to see the experimental results, I trained the model 
with a small data set. I used data augmentation in this project, so that I can get 
a robust model.
If you want to use this project in your production environment, you should **do more**.
## Train
```python train.py```
## Predict
```python predict.py```
## Web API
>http://127.0.0.1:8080/test

## Dependencies
    Python 3.5
    flask 1.1.1
    h5py 2.10.0
    Keras 2.3.1
    numpy 1.17.1
    scipy 1.3.2
    tensorflow 1.14.0
    
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

1. https://github.com/xiangrufan/tensorflow.keras-mtcnn
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

#Further Reading

1. Papers list: there are some papers that I think important.
    >https://github.com/wotchin/paper-list
2. InsightFace: some implementations of state-of-the-art face recognition algorithm.
    >https://github.com/deepinsight/insightface
3. wider face results: we can see state-of-the-art model for wider face testing dataset.
    >http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html
4. 