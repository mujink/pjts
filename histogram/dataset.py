import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

class Dataset(object):
    def __init__(self):
        self.SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__)) # c:\Users\ai\PycharmProjects\group_proj
        self.DEFAULT_DATA_DIR = os.path.join(self.SCRIPT_PATH, 'image-data')
        self.DEFAULT_LABEL_FILE = os.path.join(self.SCRIPT_PATH, './labels/2368-common-hangul.txt')
        self.IMAGE_WIDTH = 64
        self.IMAGE_HEIGHT = 64
        self.num_image = 340992
        self.csv_path = self.DEFAULT_DATA_DIR + '/labels-map.text'
        self.num_class = len(self.read_class_names())
        self.data = self.datas()
        self.batch_size = 100
        self.batch_count = 0
        self.num_samples = len(self.data)
        self.num_batchs = int(np.ceil(self.num_samples) / self.batch_size)

    def read_class_names(self):
        names = {}
        with open(self.DEFAULT_LABEL_FILE, 'r', encoding="utf-8") as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names


    def datas(self):
        if not os.path.isfile('./hangul_number_dict.picl'):
            # key 한글 : value 숫자 대응하는 딕셔너리
            hangul_number = {} 
            common_hangul = open(self.DEFAULT_LABEL_FILE, "r", encoding='utf-8')
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

        datas = []
        with open(self.csv_path, encoding="utf-8") as fd:
            y = fd.readlines()
            for line in y:
                y_label = np.zeros(self.num_class, dtype = np.float)
                line = line.strip().split(",")
                image_path = line[0]
                if line[1] == "":
                        line[1] = ","
                labels = hangul_number[line[1]]
                y_label[labels] = 1
                datas.append([image_path, y_label])
        np.random.shuffle(datas)
        return datas
    def __iter__(self):
        return self
        
    def __next__(self):
        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                train_images = np.empty((self.batch_size, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1))
                train_labels = np.empty((self.num_image), dtype=int)
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples:
                    index -= self.num_samples
                image, labels = self.data[index]
                image = cv2.imread(image)
                ret, image = cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO_INV)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                image = np.array(image).reshape(64,64,1)
                # img = image.load_img(line[0], target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale")
                img = image.astype('float32')/255.
                train_images[num,:,:,:] = img
                num += 1
            return (train_images, labels)
        else:
            self.batch_count = 0
            # np.random.shuffle(self.annotations)
            raise StopIteration
        print("[info] train_images, train_labels 완료")

        return train_images, train_labels 