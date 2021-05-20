import pandas as pd
import cv2
import numpy as np
from pandas.core.tools.datetimes import DatetimeScalar
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense, Input, BatchNormalization, GlobalMaxPool2D
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.utils import shuffle



def Recognition_model():
    input = Input(shape=(64,64,3))
    x = Conv2D(filters=(64), kernel_size=(64,3),strides=1)(input)
    x = BatchNormalization()(x)
    x1 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(3,64),strides=1)(input)
    x = BatchNormalization()(x)
    x2 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(40,40),strides=1)(input)
    x = BatchNormalization()(x)
    x3 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(5,5),strides=1)(input)
    x = BatchNormalization()(x)
    x4 = GlobalMaxPool2D()(x)

    x5 = Flatten()(x1+x2+x3+x4)
    # x5 = x1+x2+x3+x4

    x = Dense(1512)(x5)
    output = Dense(2368, activation="softmax")(x)

    model = Model(inputs=input, outputs= output)

    model.summary()
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    return model
model = Recognition_model()

path = "D:\python\pjt_odo\image-data_2/labels-map_test.text"
load = pd.read_table(path, header=None)
load = shuffle(load)

# for i in range(len(load)):
#     image_path.append(load.iloc[i,0].split(",")[0])
#     label.append(load.iloc[i,0].split(",")[1])

# print(len(image_path))
# print(len(label))

for i in range(0,16):
    step = 45288
    image_path = []
    label = []
    # 인덱스, 시작 인덱스 , 끝 인덱스
    for index in range(step*i,step*(i+1)):
        image_path.append(load.iloc[index,0].split(",")[0])
        label.append(load.iloc[index,0].split(",")[1])

    print(len(image_path))
    print(len(label))

    x_images = []
    for j, image_ph in enumerate(image_path) :
        img = cv2.imread(image_ph)
        x_images.append(img)

    x_images = np.array(x_images)
    y_label = np.array(label)

    y_label = y_label.reshape(-1,1)
    onthot = OneHotEncoder()
    onthot.fit(y_label)
    y_label = onthot.transform(y_label).toarray()
    x_images = x_images/255.

    kfold = KFold(n_splits=5, shuffle=True)
    a = 1
    for train_index, test_index in kfold.split(x_images):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        x_train, x_val = x_images[train_index], x_images[test_index]
        y_train, y_val = y_label[train_index], y_label[test_index]

        model.fit(x=x_train, y=y_train, batch_size=2350, epochs=10, validation_data=(x_val, y_val), shuffle=True)
        if a == 5:
            model.save("model4_{}.h5".format(a))
        a += 1
