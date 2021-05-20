#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.text                      = edict()
__C.text.CLASSES              = ".\labels/2368-common-hangul.txt"
__C.text.ANCHORS              = [13,31, 17,35, 17,42, 21,49, 25,50, 25,58, 30,69, 35,70, 40,81]
__C.text.STRIDES              = [8, 16, 32]
__C.text.XYSCALE              = [1.2, 1.1, 1.05]
__C.text.ANCHOR_PER_SCALE     = 3
__C.text.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.IMAGE_PATH          = ".\Test_image\images"
__C.TRAIN.LABEL_PATH          = ".\Test_image\label"
__C.TRAIN.BATCH_SIZE          = 1
__C.TRAIN.INPUT_SIZE          = (1280,1920)
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30


