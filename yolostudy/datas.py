#! /usr/bin/env python
# coding=utf-8
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import utils as utils
from config import cfg

class Dataset(object):
    """implement Dataset here"""

    def __init__(self):

        self.strides, self.anchors, self.num_classes, _ = utils.load_config()
        self.img_path = cfg.TRAIN.IMAGE_PATH
        self.label_path = cfg.TRAIN.LABEL_PATH
        self.batch_size = cfg.TRAIN.BATCH_SIZE 
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.text.CLASSES)
        self.anchor_per_scale = cfg.text.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 2400
        # 최대 박스 스케일 설정하기
        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self):
        image_path = glob.glob(os.path.join(self.img_path, '*.jpeg'))
        txt = glob.glob(os.path.join(self.label_path, '*.text'))
        annotations = []

        class_list = []
        cls = open(cfg.text.CLASSES, encoding="utf-8")
        for index, string in enumerate(cls):
            clss = string.strip()
            class_list.append(clss)
            
        for i in range(len(image_path)):
            with open(txt[i], encoding="utf-8") as fd:
                y = fd.readlines()
                string = ""
                for line in y:
                    box = line.strip()
                    box = box.split(",")
                    if box[4] == "":
                        box[4] = ","
                    class_num = int(self.class_to_number(box[4], class_list))
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    string += " {},{},{},{},{}".format(
                        x1,
                        y1,
                        x2,
                        y2,
                        class_num,
                    )
                annotations.append(image_path[i] + string)

        np.random.shuffle(annotations)
        return annotations

    def class_to_number(self, cls_str, listset):
        for i , string in enumerate(listset):
            if cls_str == string :
                return i
            elif cls_str == "," :
                return 2355

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            # self.train_output_sizes = self.train_input_size // self.strides
            self.train_output1_sizes = self.train_input_size[0] // self.strides
            self.train_output2_sizes = self.train_input_size[1] // self.strides
            batch_image = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size[0],
                    self.train_input_size[1],
                    3,
                ),
                dtype=np.float32,
            )

            batch_label_sbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output1_sizes[0],
                    self.train_output2_sizes[0],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_mbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output1_sizes[1],
                    self.train_output2_sizes[1],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )
            batch_label_lbbox = np.zeros(
                (
                    self.batch_size,
                    self.train_output1_sizes[2],
                    self.train_output2_sizes[2],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                ),
                dtype=np.float32,
            )

            batch_sbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_mbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
            )
            batch_lbboxes = np.zeros(
                (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
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
                        label_sbbox,
                        label_mbbox,
                        label_lbbox,
                        sbboxes,
                        mbboxes,
                        lbboxes,
                    ) = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return (
                    batch_image,
                    (
                        batch_smaller_target,
                        batch_medium_target,
                        batch_larger_target,
                    ),
                )
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration


    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        
        height, width, _ = image.shape
        bboxes = np.array(
            [list(map(float, box.split(","))) for box in line[1:]]
        )
        # bboxes = bboxes * np.array([width, height, width, height, 1])
        bboxes = bboxes.astype(np.int64)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size[0], self.train_input_size[1]],
            np.copy(bboxes),
        )
        return image, bboxes


    def preprocess_true_boxes(self, bboxes):
        # 불러온 데이터 셋에서 모델에 맞게 박스를 지정하여 반환
        label = [
            np.zeros(
                (
                    self.train_output1_sizes[i],
                    self.train_output2_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )

            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )
            iou = []
            exist_positive = False
            for i in range(3):
                # 트루 박스의 갯수 = 3
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][xind, yind, iou_mask, :] = 0
                    label[i][xind, yind, iou_mask, 0:4] = bbox_xywh
                    label[i][xind, yind, iou_mask, 4:5] = 1.0
                    label[i][xind, yind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)
                # print("============================")
                # print("1번",label[0].shape)
                # print("2번",label[1].shape)
                # print("3번",label[2].shape)
                # print("베스트 디텤",best_detect)
                # print("yind",yind)
                # print("xind",xind)
                # print("베스트 앵커",best_anchor)
                label[best_detect][xind-1, yind-1, best_anchor, :] = 0
                label[best_detect][xind-1, yind-1, best_anchor, 0:4] = bbox_xywh
                label[best_detect][xind-1, yind-1, best_anchor, 4:5] = 1.0
                label[best_detect][xind-1, yind-1, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
