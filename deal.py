from PIL import Image
from numpy import *
import os

path='/Users/tongshijia/Downloads/data/pic/1'
newPath='/Users/tongshijia/Downloads/data/pic/01'
for pa in os.listdir(path):
    pil_im=Image.open(os.path.join(path,pa))
    pil_im=pil_im.rotate(180)
    pil_im.save(os.path.join(newPath,pa))