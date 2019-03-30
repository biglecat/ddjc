# -*- coding: utf-8 -*-

import cv2
import numpy as np

imagedata = cv2.imread(r"F:\ddjc_new\TrainModel\testset\0\0.png")

cv2.imshow('',imagedata)
cv2.waitKey(0)

imgInfo = imagedata.shape
height = imgInfo[0]
width = imgInfo[1]
deep = imgInfo[2]
dst = np.zeros([height * 2, width, deep], np.uint8)
for i in range(height):
    for j in range(width):
        dst[i, j] = imagedata[i, j]
        dst[height * 2 - i - 1, j] = imagedata[i, j]
dstcut = dst[height:height * 2, 0:width]

cv2.imshow('dstcut',dstcut)
cv2.waitKey(0)