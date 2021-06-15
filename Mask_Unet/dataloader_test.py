from numpy.core.fromnumeric import shape
from tensorflow.python.ops.gen_math_ops import xlog1py
from dataloader import Dataset
import cv2
import numpy as np

trainset = Dataset()
green = (0,255,0)
epochs = 8
for epoch in range(epochs):
    a = 0
    for image_data, target in trainset:
        a += 1
        bach = (len(image_data))
        for i in range(bach):
            img = image_data[i]
            mask = target[i]
            print(mask.shape)
            # cv2.imshow("img", img)
            # cv2.imshow("mask_img", mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
