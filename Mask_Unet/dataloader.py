#! /usr/bin/env python
# coding=utf-8
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import utils as utils
from config import cfg
import xml.etree.ElementTree as Et
import random
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical

class Dataset(object):
    """implement Dataset here"""

    def __init__(self):

        self.img_path = cfg.TRAIN.IMAGE_PATH
        self.label_path = cfg.TRAIN.LABEL_PATH
        self.batch_size = cfg.TRAIN.BATCH_SIZE 
        self.train_input_size = cfg.TRAIN.INPUT_SIZE

        self.classstext = self.class_num(self.label_path)
        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.data_aug = cfg.TRAIN.DATA_AUG
        self.num_classes = len(self.classstext)

    def class_num(self, ann):
        lst_class = []
        lst = {}
        ann_root, ann_dir, ann_files = next(os.walk(ann))
        for text_file in ann_files:
            text = open(os.path.join(ann_root, text_file), "r", encoding="utf-8")
            text = text.readlines()
            for line in text:
                value = line.strip().split(",")
                typ = str(value[8])
                name = str(value[9])
                if typ == "Korean" and name is not ["###", None]:
                    lst_class.append(name)
        lst_class = set(lst_class)
        for i, tx in enumerate(lst_class):
            lst[tx] = i
        return lst

    def load_annotations(self):

        ann_root, ann_dir, ann_files = next(os.walk(self.label_path))
        img_root, amg_dir, img_files = next(os.walk(self.img_path))
        annotations = []

        for text_file in ann_files:
            img_name = img_files[img_files.index(".".join([text_file.split(".")[0], "jpg"]))]
            image_path = os.path.join(img_root, img_name)

            text = open(os.path.join(ann_root, text_file), "r", encoding="utf-8")
            text = text.readlines()
            string = ""
            for line in text:
                value = line.strip().split(",")

                left_up_x = float(value[0]) 
                left_up_y = float(value[1])
                right_up_x = float(value[2])
                right_up_y = float(value[3])
                right_donw_x = float(value[4])
                right_donw_y = float(value[5])
                left_donw_x = float(value[6])
                left_donw_y = float(value[7])
                typ = str(value[8])
                name = str(value[9])
                lst = self.classstext
                if typ == "Korean" and name != "###":
                    name = self.classstext[name]
                    string += " {},{},{},{},{},{},{},{},{}".format(
                        left_up_x,
                        left_up_y,
                        right_up_x,
                        right_up_y,
                        right_donw_x,
                        right_donw_y,
                        left_donw_x,
                        left_donw_y,
                        name,
                    )
                    
            if len(string) < 1:
                print(len(string))
                continue

            annotations.append(image_path + string)
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):

            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3,
                ),
                dtype=np.float32,
            )
            batch_mask = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3
                ),
                dtype=np.float32,
            )
            batch_path = np.zeros(
                (
                    self.batch_size,
                    1
                ),
                dtype=np.str,
            )
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    
                    image, bboxes = self.parse_annotation(annotation)
                    (   
                        mask,
                        label
                    ) = self.preprocess_true_boxes(bboxes)
                    batch_image[num, :, :, :] = image
                    batch_mask[num, :, :, :] = mask
                    num += 1
                self.batch_count += 1 

                return (batch_image,
                        batch_mask)
            else:
                self.batch_count = 0
                # np.random.shuffle(self.annotations)
                raise StopIteration
 
    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)

        bboxes = np.array(
            [list(map(float, box.split(","))) for box in line[1:]]
        )
        # bboxes = bboxes * np.array([width, height, width, height, 1])
        bboxes = bboxes.astype(np.int64)
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )

        return image, bboxes


    def preprocess_true_boxes(self, bboxes):
        # 불러온 데이터 셋에서 모델에 맞게 박스를 지정하여 반환
        mask = np.zeros([self.train_input_size, self.train_input_size, 3], np.uint8)
        label = []
        kernel = np.ones((3,3), np.uint8)
        for bbox in bboxes:
            bbox_coor = bbox[:8]
            bbox_class_ind = bbox[8]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )

            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            label.append(smooth_onehot)
            pts = np.array([[bbox_coor[6],bbox_coor[7]],
                            [bbox_coor[4],bbox_coor[5]],
                            [bbox_coor[2],bbox_coor[3]],
                            [bbox_coor[0],bbox_coor[1]]]).astype(np.int32)
                            
            mask = cv2.drawContours(mask, [pts], -1, (1, 1, 1), -1, cv2.LINE_AA)
        # cv2.imshow("mask_img", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # mask = cv2.erode(mask, kernel, iterations=2)
        # cv2.imshow("mask_img", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return mask, label

    def __len__(self):
        return self.num_batchs
