# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
# import os
#
#
# train_dir = r'F:\tra\pic\train'
# # train_normal_dir = os.path.join(train_dir,'normals')
# # trian_broken_dir = os.path.join(train_dir,'brokens')
#
#
# #将所有图片乘1/255缩放
# train_datagen = ImageDataGenerator(rescale=1./255)
#
# #创建python生成器
# train_generator = train_datagen.flow_from_directory(train_dir,
#                                                     target_size=(100,100),
#                                                     batch_size=20,
#                                                     class_mode='binary')

# for data_batch,label_batch in train_generator:
#     print('data batch shape:',data_batch.shape)
#     print('labels batch shape:',label_batch.shape)
#     print('label: ',label_batch)
#     break


from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2

data_gen = ImageDataGenerator(rotation_range=(0-60))
imagespath = r'F:\AA\0308\ddjc_march\TestVersion\Test\oldpic\data\train\0'
respath = r"F:\AA\0308\ddjc_march\TestVersion\Test\respicmed\0"

imglist = os.listdir(imagespath)

for name in imglist:

    imagepath = os.path.join(imagespath,name)
    #读取图片并调整大小
    imgdata = image.load_img(imagepath,target_size=(100,100))
    #转换为numpy数组
    x = image.img_to_array(imgdata)
    #改变形状 为(1,100,100,3)
    x = x.reshape((1,)+x.shape)

    i = 0
    for batch in data_gen.flow(x,batch_size=1):
        plt.figure(1)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i+=1
        if i%4 == 0:
            img = image.array_to_img(batch[0])
            img.save(os.path.join(respath,name))
            break

    # plt.show()






















