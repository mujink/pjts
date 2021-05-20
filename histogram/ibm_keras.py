import os
import cv2
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D,Reshape, Bidirectional, LSTM, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


##### <CONSTANTS> #####
""" 
Prerequisite: Run img_gen.py file in tools subdirectory. AND
You should have 2350-common-hangul.txt file in the specifed path below
"""
# Default paths.
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__)) # c:\Users\ai\PycharmProjects\group_proj

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2368-common-hangul.txt')
DEFAULT_DATA_DIR = os.path.join(SCRIPT_PATH, 'image-data')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_PATH, 'saved-model')
DEFAULT_NPY_DIR = os.path.join(SCRIPT_PATH, 'npy')

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_NUM_EPOCHS = 300
BATCH_SIZE = 200

# This will be determined by the number of entries in the given label file.
num_image = 340992
num_classes = 2368
keep_prob = 0.5
##### </CONSTANTS> #####

def make_data(csv_path):
       if not os.path.isfile('./hangul_number_dict.picl'):
              # key 한글 : value 숫자 대응하는 딕셔너리
              hangul_number = {} 
              common_hangul = open(DEFAULT_LABEL_FILE, "r", encoding='utf-8')
              i = 0
              while True:
                     hangul = common_hangul.readline().strip()
                     if hangul == "":
                            hangul = ","
                     hangul_number[str(hangul)] = i
                     i += 1

              common_hangul.close()

              # 딕셔너리 pickle로 저장하기
              try:
                     file_handler = open('hangul_number_dict.picl', 'wb')
                     pickle.dump(hangul_number, file_handler)
                     file_handler.close()
                     print("[info] 한글 대 숫자 대응 딕셔너리 완성")
              except:
                     print("Something went wrong")
       else:
              file_handler = open("hangul_number_dict.picl", "rb")
              hangul_number = pickle.load(file_handler)
              file_handler.close()

       # 판다스로 csv파일을 읽어옴
       # df = pd.read_csv(csv_path, header=None, encoding="uft-8", sep=",") # [num_image rows x 2 columns]
       
       train_images = np.empty((num_image, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
       train_labels = np.empty((num_image), dtype=int)
       with open(csv_path, encoding="utf-8") as fd:
                y = fd.readlines()
                np.random.shuffle(y)
                for idx, line in enumerate(y):
                     line = line.strip().split(",")
                     img = image.load_img(line[0], target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale")
                     img = image.img_to_array(img).astype('float32')/255.
                     train_images[idx,:,:,:] = img
                     if line[1] == "":
                            line[1] = ","
                     ganada = line[1]
                     train_labels[idx] = hangul_number[ganada]

       # df에 있는 각 경로에 대해서 반복하여

       train_labels = to_categorical(train_labels) # (num_image, 2350)
       print("[info] train_images, train_labels 완료")

       return train_images, train_labels

def make_model(train_images, train_labels):
       # Create the model!
       model = Sequential([
       # First convolutional layer. 32 feature maps.
       Conv2D(filters=32, kernel_size=5,
              strides=(1, 1), padding='same',
              activation='relu',
              input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),

       # Second convolutional layer. 64 feature maps.
       Conv2D(filters=64, kernel_size=5,
              strides=(1, 1), padding='same',
              activation='relu'),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),
       
       # Third convolutional layer. 128 feature maps.
       Conv2D(filters=128, kernel_size=3,
              strides=(1, 1), padding='same',
              activation='relu'),
       MaxPool2D(pool_size=(2, 2), strides=2,
              padding='same'),
       
       # Fully connected layer. Here we choose to have 1024 neurons in this layer.
       Flatten(),
       Dense(units=2500, activation='relu'),

       # Dropout layer. This helps fight overfitting.
       Dropout(rate=keep_prob),

       # Classification layer.
       Dense(units=num_classes, activation='softmax')
       ])

       er = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
       re = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1)

       model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
       model.fit(x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=DEFAULT_NUM_EPOCHS, callbacks=[er, re], validation_split=0.2, shuffle=True)
       return model

def make_model2(train_images, train_labels):
       # Create the model!
       input = Input(shape=(IMAGE_WIDTH,IMAGE_WIDTH,1))
       x = Conv2D(filters=(128), kernel_size=(3,3),strides=1, activation="swish")(input)
       x = BatchNormalization()(x)
       # x1 = MaxPool2D()(x)

       x = Conv2D(filters=(64), kernel_size=(2,2),strides=1,activation="swish")(x)
       x = BatchNormalization()(x)
       # x2 = MaxPool2D()(x)

       x = Conv2D(filters=(128), kernel_size=(4,4),strides=1,activation="swish")(x)
       x = BatchNormalization()(x)
       # x3 = MaxPool2D()(x)

       x = Conv2D(filters=(64), kernel_size=(5,5),strides=1,activation="swish")(x)
       x = BatchNormalization()(x)
       x = MaxPool2D()(x)
       # x = Reshape((27*27,-1))(x)
       # x = Bidirectional(LSTM(64, activation="tanh"))(x)
       # # x5 = (x1+x2+x3+x4)
       x = Flatten()(x)
       x = Dense(2500)(x)
       x = Dropout(rate=keep_prob)(x)
       output = Dense(2368, activation="softmax")(x)

       model = Model(inputs=input, outputs= output)

       er = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
       re = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1)

       model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
       model.fit(x=train_images, y=train_labels, batch_size=BATCH_SIZE, epochs=DEFAULT_NUM_EPOCHS, callbacks=[er, re], validation_split=0.2, shuffle=True)
       return model


if __name__ == '__main__':
       # train_images, train_labels = make_data(os.path.join(DEFAULT_DATA_DIR, '/labels-map.csv'))
       # train_images, train_labels = make_data(DEFAULT_DATA_DIR + '/labels-map.text')
       # np.save(os.path.join(DEFAULT_NPY_DIR, 'train_images.npy'), arr=train_images)
       # np.save(os.path.join(DEFAULT_NPY_DIR, 'train_labels.npy'), arr=train_labels)
       train_images = np.load(os.path.join(DEFAULT_NPY_DIR, 'train_images.npy'))
       train_labels = np.load(os.path.join(DEFAULT_NPY_DIR, 'train_labels.npy'))       
       # model = make_model(train_images, train_labels)
       # if not os.path.exists(DEFAULT_MODEL_DIR):
       #        os.makedirs(DEFAULT_MODEL_DIR)
       # model.save(os.path.join(DEFAULT_MODEL_DIR, 'mymodel.h5'))
       model = make_model2(train_images, train_labels)
       model.save(os.path.join(DEFAULT_MODEL_DIR, 'mymodel4.h5'))

