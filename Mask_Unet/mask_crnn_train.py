# 파일
from models import Unet, initlayer, CRNN
import utils
from config import cfg
import numpy as np

# 텐서플로
import tensorflow as tf

# 데이터 전처리
from dataloader import Dataset

# 모델
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# 디코드

# 옵티마이저
from tensorflow.keras.optimizers import Adam

# 로스
# from loss import compute_loss

# 텐서보드 로그 쓰기
import os
import shutil

from absl import app, flags

def main(_argv):
    
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logdir = "./data/log"
    isfreeze = False
    

    # dataset read
    trainset = Dataset()
    num_classes = 5071
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
    input = Input(shape=(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3))
    output = Unet(input)
    y_pred = CRNN(output, num_classes)
    #모델을 정의 : Model(input, output)

    
    model = Model(inputs=input, outputs= y_pred)

    
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

            # optimizing process
            total_loss = tf.keras.losses.binary_crossentropy(target,pred_result)

            # 모델의 변수와 로스를 넣어 그라디언트를 구함
            gradients = tape.gradient(total_loss, model.trainable_variables)

            # 그라디언트를 적용함
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss = tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1,2]))

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
        for image_data, target in trainset:
            train_step(image_data, target)

        model.save("./checkpoints/segmentation_model2.h5")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

    
    
    # train모델이아닌 preiction 모델의 output은 softmax activation을 적용한 값
    if prediction_only:
        return model_pred

    #최대 글자수
    max_string_len = int(y_pred.shape[1])

    # CTC LOSS 함수 정의
    def ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    # CTC LOSS를 계산할때 사용하는 INPUT 정의
    labels = Input(name='label_input', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Lambda를 사용하여 ctc loss 구한다
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])
    
    # 최종 학습모델의 인풋은 4가지이고, 아웃풋은 ctc loss 값
    model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=ctc_loss)
    
    return model_train, model_pred