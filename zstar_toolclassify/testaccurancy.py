# -*- coding: utf-8 -*-

from keras.models import load_model
import cv2
import numpy as np
import os

model=load_model(r'F:\tra\_kerasmodel\ImageClassify\zstar_toolclassify\tool\sweetmodel_newtrainset_20.h5')

imgspath = r'F:\AA\0308\ddjc_march\TestVersion\Test\testimg\0'

def prob_1(imgspath):
    normalsize = 0
    brokensize = 0

    imglist = os.listdir(imgspath)
    sum = len(imglist)

    for imgname in imglist:
        imgpath = os.path.join(imgspath,imgname)
        img = cv2.imread(imgpath)
        img = cv2.resize(img,(100,100))
        # 转为训练的数据格式
        resimg = []
        resimg.append(img)
        resimg = np.array(resimg) / 255
        pre_y = model.predict(resimg)

        res = pre_y[0][0]

        if res < 0.5:
            normalsize+=1
        else:
            brokensize+=1

    print('nor% : ',float(normalsize)/float(sum))
    print('bro% : ',float(brokensize)/float(sum))



if __name__ == '__main__':
    prob_1(imgspath)







