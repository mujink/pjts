import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import utils
import cv2
import numpy as np
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.keras.models import load_model
import io, os
from config import cfg
from matplotlib import pyplot as plt
def main(_args):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = cfg.TRAIN.INPUT_SIZE
    # image_path = "D:\python\pjt_odo\ImagesPart2/tr_img_05350.jpg"
    image_path = "D:\python\pjt_odo\ImagesPart2/tr_img_05030.jpg"
    image = cv2.imread(image_path)

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_datas = utils.image_preprocess(np.copy(original_image), [input_size, input_size])

    images_data = []
    for i in range(1):
        images_data.append(image_datas)
    images_data = np.asarray(images_data).astype(np.float32)
    batch_data = tf.constant(images_data)


    # model = load_model("./checkpoints/segmentation_model_ori.h5")
    model = load_model("./checkpoints/segmentation_model5.h5")
    # model.summary()
    result = model.predict(batch_data)
    # result = (model.predict(batch_data)[0,:,:,:] > 0.5).astype(np.uint8)
    result = np.array(result).reshape(input_size, input_size)
    plt.imshow(result, cmap="gray")

    # result = np.array(result).reshape(input_size, input_size, 3)
    cv2.imshow("img",image_datas)
    # cv2.imshow("result",result*255)
    cv2.imshow("result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass