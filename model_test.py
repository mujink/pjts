from keras.models import load_model
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
path = "D:\python\pjt_odo\image-data_test\labels-map_test.csv"
load = pd.read_csv(path,encoding="utf-8",error_bad_lines=False,quotechar=None, quoting=3)

lst_image_path = list(load.iloc[:,0])
lst_image_lable = list(load.iloc[:,-1])


x_images = []
for i, image_path in enumerate(lst_image_path) :
    print(i)
    img = cv2.imread(image_path)
    x_images.append(img)
x_images = np.array(x_images, dtype="f2")
y_label = np.array(lst_image_lable)
x_images = x_images/255.


print(y_label.shape)
print(x_images.shape)

y_label = y_label.reshape(-1,1)
onthot = OneHotEncoder()
onthot.fit(y_label)
y_label = onthot.transform(y_label).toarray()


# for i in range(10):
model = load_model("model1_42299.h5")
y_pred = model.evaluate(x_images,y_label)

print(y_pred)
