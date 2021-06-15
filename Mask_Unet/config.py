#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# Train options
__C.TRAIN                     = edict()
__C.TRAIN.IMAGE_PATH          = "D:\python\pjt_odo\ImagesPart2"
__C.TRAIN.LABEL_PATH          = "D:\python\pjt_odo/train_gt_t13"
__C.TRAIN.BATCH_SIZE          = 10
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = False

__C.TRAIN.LR_INIT             = 1e-1
__C.TRAIN.LR_END              = 1e-3

__C.TRAIN.WARMUP_EPOCHS       = 3
__C.TRAIN.FISRT_STAGE_EPOCHS    = 10
__C.TRAIN.SECOND_STAGE_EPOCHS   = 20


