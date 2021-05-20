# 파일
from models import YOLOv4, initlayer
import utils
from config import cfg
import numpy as np

# 텐서플로
import tensorflow as tf

# 데이터 전처리
from datas import Dataset

# 모델
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# 디코드
from decode import decode_train

# 옵티마이저
from tensorflow.keras.optimizers import Adam

# 로스
from loss import compute_loss

# 텐서보드 로그 쓰기
import os
import shutil

from absl import app, flags

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

def main(_argv):
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logdir = "./data/log"
    isfreeze = False
    
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    IOU_LOSS_THRESH = cfg.text.IOU_LOSS_THRESH

    # dataset read
    trainset = Dataset()

    # epoch
    steps_per_epoch = len(trainset) # 13121
    print(steps_per_epoch)

    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS

    # steps
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    # optimizer
    optimizer = Adam()

    # model load
    input = Input(shape=(cfg.TRAIN.INPUT_SIZE[0], cfg.TRAIN.INPUT_SIZE[1], 3))
    output = YOLOv4(input, NUM_CLASS)
    bbox_tensors = []
    x = cfg.TRAIN.INPUT_SIZE[0]
    y = cfg.TRAIN.INPUT_SIZE[1]
    for i, fm in enumerate(output):
        if i == 0:
            bbox_tensor = decode_train(fm, (x//32), (y//32), NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
            bbox_tensor = decode_train(fm,  (x//64), (y//64), NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            bbox_tensor = decode_train(fm, (x//128),(y//128), NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    model = Model(inputs=input, outputs= bbox_tensors)
    model.summary()

    
    # log writer
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        # 그라디언트 수행
        with tf.GradientTape() as tape:
            # 출력
            pred_result = model(image_data, training=True)
            # print(len(pred_result)) = 6
            giou_loss = 0
            conf_loss = 0
            prob_loss = 0
            # optimizing process
            for i in range(3):
                # 각 앵커마다 출력 값을 받고 작성된 로스 함수로 로스를 구함.
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                # print("conv_shape", conv.shape)           = (1, 212, 308, 7119)
                # print("pred_shape", pred.shape)           = (1, 212, 308, 3, 2373)
                # print("target_shape", target[i][0].shape) = (1, 212, 308, 3, 2373)
                # print("bbox_shape", target[i][1].shape)   = (1, 2400, 4)
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            # 모델의 변수와 로스를 넣어 그라디언트를 구함
            gradients = tape.gradient(total_loss, model.trainable_variables)
            # 그라디언트를 적용함
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    for epoch in range(first_stage_epochs + second_stage_epochs):

        for image_data, target in trainset:
            train_step(image_data, target)

        model.save("./checkpoints/model.h5")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass