from matplotlib.pyplot import box
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession
from tensorflow.keras.models import load_model
import io, os

flags.DEFINE_integer('size', 224, 'resize images to')
flags.DEFINE_string('output', 'result.png', 'path to output image')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 224
    image_path = "./dataset\dataset/test/test_5.jpg"
    # image_path = "./dataset\dataset/raccoon_images/raccoon-3.jpg"

    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_datas = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    # image_data = cv2.resize(original_image, (input_size, input_size))
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_datas)
    images_data = np.asarray(images_data).astype(np.float32)

    model = load_model("./checkpoints/model6.h5")
    # saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    # infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    result = model.predict(batch_data)

    boxes =[]
    pred_conf = []    
    # for value in result:
    x = input_size
    scale = [8,16,32]
    for i in range(3):
        conv, pred = result[i * 2], result[i * 2 + 1]
        print(pred.shape)
        boxes.append(pred[..., :6])

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in boxes]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.1)
    bboxes = utils.nms(bboxes, 0.215, method='soft-nms')

    """
    DEFAULT_OUTPUT_DIR = "raccon_prd_bbox"
    labels_txt = io.open(os.path.join(DEFAULT_OUTPUT_DIR, '{}.text'.format("raccoon-1")), 'w',
                    encoding='utf-8')
    

    for i, value in enumerate(bboxes):
        x1 = value[0]
        y1 = value[1]
        x2 = value[2]
        y2 = value[3]
        # cof = value[4]

        labels_txt.write(u'{},{},{},{}\n'.format(int(x1), int(y1), int(x2), int(y2)))
        # labels_txt.write(u'{},{},{},{},{}\n'.format(int(x1), int(y1), int(x2), int(y2), int(cof)))
    labels_txt.close()"""
    green = (0,255,0)
    rad = (0,0,255)
    a = 0
    bbox_thick = int(0.6 * (input_size + input_size) / 600)
    for i, value in enumerate(bboxes):
        print(value.shape)
        print(value[0])
        print(value[1])
        print(value[2])
        print(value[3])
        print("신뢰도",value[4])
        print("클래스",value[5])
        if value[4] > 0.1 :
            if value[5] <= 0.0 :
                num = "raccon"
            else:
                num = "?????"
            bbox_mess = '%s: %.2f' % (num, value[4])
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=bbox_thick // 2)[0]
            c3 = (value[0] + t_size[0], value[1] - t_size[1] - 3)
            image = cv2.rectangle(image, (int(value[0]), int(value[1])), (np.float32(c3[0]), np.float32(c3[1])), green, -1) #filled
            img = cv2.rectangle(image,(int(value[0]), int(value[1])),(int(value[2]), int(value[3])), green ,2)
            cv2.putText(img,  bbox_mess, (int(value[0]), int(value[1])- 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            cv2.imshow("img",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("./dataset\dataset/test/test_5_{}.jpg".format(a), img)
            a += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
