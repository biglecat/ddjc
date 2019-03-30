# # -*- coding: utf-8 -*-
#
# import numpy as np
#
# import argparse
#
# import imutils
#
# import cv2
#
# # load the image from disk
#
# image = cv2.imread(r"F:\ddjc_new\TrainModel\testset\0\3.jpg")
#
# # loop over the rotation angles
#
# for angle in np.arange(0, 360, 15):
#
#     rotated = imutils.rotate(image, angle)
#
#     cv2.imshow("Rotated (Problematic)", rotated)
#
#     cv2.waitKey(0)
#
# # loop over the rotation angles again, this time ensure the
#
# # entire pill is still within the ROI after rotation
#
# for angle in np.arange(0, 360, 15):
#
#     rotated = imutils.rotate_bound(image, angle)
#
#     cv2.imshow("Rotated (Correct)", rotated)
#
#     cv2.waitKey(0)



import cv2
import math

def rotate(img, angle):
    height = img.shape[0]
    width = img.shape[1]

    if angle % 180 == 0:
        scale = 1
    elif angle % 90 == 0:
        scale = float(max(height, width)) / min(height, width)
    else:
        scale = math.sqrt(pow(height, 2) + pow(width, 2)) / min(height, width)

    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))

    return rotateImg

if __name__ == '__main__':
    image = cv2.imread(r"F:\ddjc_new\TrainModel\testset\0\8.png")
    i = 30
    while i<=360:
        img = rotate(image,i)
        cv2.imshow("Rotated", img)
        cv2.waitKey(0)
        i+=30