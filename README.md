# SmooFaceEngine
An open source face recognition engine named SmooFaceEngine.

Author: @wotchin
# Introduction
Let us see something about this project now.
## What is this project?
This project is an open source project about
 face recognition. In the project, we implement a face 
 recognition engine that is one-shot training and could predict
 unknown faces.

## Why did this?
JUST FOR FUN!

# Principle of SmooFaceNet
We use deep CNN as the basic network. In this project,
 we implemented some CNN, such as VGGNet, VIPL face net, 
 ResNet and XCEPTION and so on.

We use AM-Softmax loss as the loss function here, rather than 
triple loss or other metric learning loss functions. This is because 
AM-Softmax has less training time, but the accuracy is still nice.

# How to use?
This project is a demo. In order to see the experimental results, I trained the model 
by a small dataset. I used data augmentation in this project, so that I can get 
a robust model.
If you want to use this project in your production environment, you should do more work.
## Train
```python train.py```
## Predict
```python predict.py```
## Web API
>http://127.0.0.1:8080/test
# Reference
## Papers


    AM-Softmax
    Sphere face
    FaceNet
    ResNet
    Xception
    VIPL Face net


(shortening， you can search them in [Google Scholar](https://scholar.google.com/))
## Open source project
waiting for updating
# LICENSE
Apache license version 2.0
# How to contribute
Just send a pull request(Merge Request) to me.
