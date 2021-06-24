# 파일
from models import Unet, initlayer, get_model
import utils
from config import cfg
import numpy as np

# 텐서플로
import tensorflow as tf
import tensorflow.keras as K
# 데이터 전처리
from dataloader import Dataset

# 모델
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# 디코드

# 옵티마이저
from tensorflow.keras.optimizers import Adam, RMSprop

# 로스
# from loss import compute_loss

# 텐서보드 로그 쓰기
import os
import shutil
import cv2

from absl import app, flags

def main(_argv):
    
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logdir = "./data/log"
    isfreeze = False
    

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
    # optimizer = RMSprop()
    # model load
    input = Input(shape=(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3))
    output = Unet(input)
    # output = get_model(input)
    
    # model = load_model("./checkpoints/segmentation_model5.h5")
    model = Model(inputs=input, outputs= output)
    # model.summary()

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
            # mask_focal = tf.reduce_sum(tf.abs(target - pred_result)*10, axis=[-1])
            # total_loss = mask_focal*tf.keras.losses.binary_crossentropy(target,pred_result)

            total_loss = tf.keras.losses.binary_crossentropy(target,pred_result)
            # 모델의 변수와 로스를 넣어 그라디언트를 구함
            gradients = tape.gradient(total_loss, model.trainable_variables)

            total_loss = tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1,2]))

            # 그라디언트를 적용함
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.print("=> STEP %4d/%4d   lr: %.6f  "
                     "total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(), total_loss))
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
            writer.flush()

    for epoch in range(first_stage_epochs + second_stage_epochs):
        for image_data, target, bathe_class in trainset:
            train_step(image_data, target)

        model.save("./checkpoints/segmentation_model5.h5")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass