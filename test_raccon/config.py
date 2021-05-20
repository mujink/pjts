#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.text                      = edict()
__C.text.CLASSES              = "./raccoon/class.txt"
__C.text.ANCHORS              = [48,91, 89,148, 131,237, 161,320, 188,335, 234,422, 263,454, 330,689, 446,877]
__C.text.STRIDES              = [8, 16, 32]
__C.text.XYSCALE              = [1.2, 1.1, 1.05]
__C.text.ANCHOR_PER_SCALE     = 3
__C.text.IOU_LOSS_THRESH      = 0.9


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.IMAGE_PATH          = "./raccoon/image"
__C.TRAIN.LABEL_PATH          = "./raccoon/xml"
__C.TRAIN.BATCH_SIZE          = 2
__C.TRAIN.INPUT_SIZE          = 224
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 100
__C.TRAIN.SECOND_STAGE_EPOCHS   = 200


