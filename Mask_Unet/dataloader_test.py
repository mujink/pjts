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
    for image_data, target, bathe_class in trainset:
        a += 1
        bach = (len(image_data))
        for i in range(bach):
            img = image_data[i]
            mask = target[i]
            label = bathe_class[i]
            cv2.imshow("img", img)
            cv2.imshow("1", mask[:,:,0])
            cv2.imshow("2", mask[:,:,1])
            cv2.imshow("3", mask[:,:,2])
            cv2.imshow("4", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()