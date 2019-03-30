# -*- coding: utf-8 -*-

from keras import  optimizers
from zstar_toolclassify import modelnet
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = r'F:\AA\0308\ddjc_march\TestVersion\Test\trainset'
TEST_DIR = r'F:\AA\0308\ddjc_march\TestVersion\Test\testimg'
IMGSIZE = (100, 100)
BATCHSIZE = 100
CLASSMODE = 'binary'

def train():
    model = modelnet.mymodel()

    #配置模型
    model.compile(loss='binary_crossentropy',
                  optimizer = optimizers.RMSprop(lr=1e-4),
                  metrics = ['acc'])

    #读取训练数据
    #配置训练数据集
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=IMGSIZE,
                                                        batch_size=BATCHSIZE,
                                                        class_mode=CLASSMODE)

    # 查看数据生成器中的数据
    # for data,label in train_generator:
    #     print(data.shape)
    #     print(label)



    #配置测试数据集
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                      target_size=IMGSIZE,
                                                      batch_size=BATCHSIZE,
                                                      class_mode=CLASSMODE)


    #利用训练数据拟合模型
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = 70,#每一轮从生成器中抽取steps_per_epoch个batchsize
                                  epochs = 20,
                                  validation_data=test_generator,
                                  validation_steps= 30)

    model.save('sweetmodel_newtrainset_20.h5')

    return history





