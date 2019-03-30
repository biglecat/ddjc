# -*- coding: utf-8 -*-

#todo 构建网络

from keras import layers
from keras import models

def mymodel():
    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.summary()

    return model

mymodel()