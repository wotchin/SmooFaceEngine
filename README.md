# SmooFaceEngine
An open-source face recognition engine.

Support for Tensorflow-2.0.0+ @branch **tensorflow2.0.0+**.

[**Further reading**](https://github.com/wotchin/paper-list/blob/master/computer-vision.md): Read more related papers.

[勘误表(errate)](https://github.com/wotchin/SmooFaceEngine/wiki/errata) 

[Chinese Wiki](https://github.com/wotchin/SmooFaceEngine/wiki)

# Introduction
## What is this project?
SmooFaceEngine: an open-source implementation for face recognition. 

In the project, SmooFaceEngine implements a face recognition engine with one-shot training.

# Principle of SmooFaceEngine
SmooFaceEngine employs several CNNs (VGGNet, VIPL face net, ResNet, XCEPTION, et al.) to recognize face images.

Firstly, SmooFaceEngine employs AM-Softmax loss as the cost function rather than triple-loss or other metric learning methods (e.g., siamese network) since AM-Softmax costs less training time and obtains higher accuracy. Although AM-Softmax is no longer a state-of-the-art model, subsequent not a few approaches follow the primary thought, e.g., [SphereFace](https://arxiv.org/abs/1704.08063).

Secondly, SmooFaceEngine uses data augmentation to generate a more robust model. By the way, some GAN approaches have broken through in this field in recent times. Readers could follow the [paper list](https://github.com/wotchin/paper-list/blob/master/computer-vision.md) above mentioned. 

Finally, SmooFaceEngine has trained a model, but this model is a classification model. Therefore, users cannot compare whether two face images are similar and know the probability. Thus, SmooFaceEngine implements a metric method, which is cosine similarity. Users could supply two images by this method, and SmooFaceEngine generates two vectors to represent the two faces. They are then using cosine similarly to measure the similar probability. 

To summarize, SmooFaceEngine cannot give the end-to-end similarity for two face images, which combines classification learning and the measuring method to evaluate similarity. AM-Softmax outperforms the softmax function in this scenario. This is why SmooFaceEngine does not directly use the softmax function as the output layer. 

# How to use it?
SmooFaceEngine is only a **demo**. SmooFaceEngine pre-trained a model with small data to see the experimental results.
If you want to use this SmooFaceEngine in your production environment directly, you should **do more**, such as training with lots of samples, training with more batches and epochs. 

## Training phase
Running the script train.py, like the following:

```python3 train.py```

## Prediction phase
Starting predict.py, then this script will validate some specified images, as follows:

```python3 predict.py```

## Web API
Meanwhile, SmooFaceEngine offers a web interface. Users could have a taste of how the engine works. 

>http://127.0.0.1:8080/test

## Dependencies

    Require Python 3.6+.
    
    Other dependencies: 
    
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
Apache License version 2.0
# How to contribute
Send pull requests or issues directly. Thank you big time :)


