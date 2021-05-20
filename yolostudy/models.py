from tensorflow.python.keras.layers.convolutional import ZeroPadding2D, Conv2D
from tensorflow.python.ops.gen_nn_ops import conv2d
from model_block import con2d, res_block, upsample
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def initlayer(input_layer):
# def initlayer(input_layer, NUM_CLASS):
    """
    input_data shape = (None, 2481,1713,3) Image
    out = feature map(feature1, feature2, feature3)

    feature1 = (None, 200, 292, 128)
    ===============================
    Total params: 3,209,824
    Trainable params: 3,201,760
    Non-trainable params: 8,064
    _______________________________
    feature2 = (None, 100, 146, 256)
    ===============================
    Total params: 14,901,856
    Trainable params: 14,880,480
    Non-trainable params: 21,376
    _______________________________
    feature3 = (None, 50, 73, 512)
    ===============================
    Total params: 24,873,568
    Trainable params: 24,847,072
    Non-trainable params: 26,496
    _______________________________
    """
    
    CBL1 = con2d(input_layer, (3, 3,  3,  32))
    resloop = con2d(CBL1, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        resloop = res_block(resloop,  64,  32, 64)

    resloop = con2d(resloop, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        resloop = res_block(resloop, 128,  64, 128)

    resloop = con2d(resloop, (3, 3, 128, 128), downsample=True)

    for i in range(8):
        resloop = res_block(resloop, 128, 64, 128)

    # resloop = con2d(resloop, (1, 1, 128, 3 * (NUM_CLASS + 5)), bn=False, downsample=True)

    feature1 = resloop
    resloop = con2d(resloop, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        resloop = res_block(resloop, 256, 128, 256)

    # resloop = con2d(resloop, (1, 1, 256, 3 * (NUM_CLASS + 5)), bn=False,downsample=True)

    feature2 = resloop
    resloop = con2d(resloop, (3, 3, 256, 512), downsample=True)

    for i in range(4):
        feature3 = res_block(resloop, 512, 256, 512)

    # feature3 = con2d(feature3, (1, 1, 512, 3 * (NUM_CLASS + 5)), bn=False,downsample=True)
    
    print(feature1.shape)
    print(feature2.shape)
    print(feature3.shape)
    return feature1, feature2, feature3

def YOLOv4(input_layer, NUM_CLASS):
    """
    ===============================
    Total params: 63,289,549
    Trainable params: 63,240,269
    Non-trainable params: 49,280
    _______________________________
    conv_sbbox = (None, 212, 308, 7119)
    ===============================
    Total params: 30,912,303
    Trainable params: 30,878,383
    Non-trainable params: 33,920
    _______________________________
    conv_mbbox = (None, 106, 154, 7119)
    ===============================
    Total params: 36,676,143
    Trainable params: 36,637,615
    Non-trainable params: 38,528
    _______________________________
    conv_lbbox = (None, 53, 77, 7119)
    ===============================
    Total params: 56,330,287
    Trainable params: 56,282,543
    Non-trainable params: 47,744
    _______________________________
    """
    feature1, feature2, feature3 = initlayer(input_layer)
    # feature1, feature2, feature3
    route = feature3
    CBL1 = con2d(feature3, (3, 3, 512, 256))
    UPS1 = upsample(CBL1)
    CBL2 = con2d(feature2, (1, 1, 512, 256))
    # CBL2_1 = Conv2D(256, 2)(CBL2)

    layer_add1 = tf.concat([CBL2, UPS1], -1)

    CBL3 = con2d(layer_add1, (1, 1, 512, 256))
    CBL4 = con2d(CBL3, (3, 3, 256, 128))
    # CBL5 = con2d(CBL4, (1, 1, 128, 256))
    CBL6 = con2d(CBL4, (3, 3, 128, 256))
    # CBL7 = con2d(CBL6, (1, 1, 128, 256))

    route_2 = CBL6
    CBL8 = con2d(CBL6, (1, 1, 256, 128))
    UPS2 = upsample(CBL8)
    CBL9 = con2d(feature1, (1, 1, 256, 128))
    # UPS2_1 = ZeroPadding2D(1)(UPS2)

    layer_add2 = tf.concat([CBL9, UPS2], -1)

    CBL11 = con2d(layer_add2, (1, 1, 256, 128))
    CBL12 = con2d(CBL11, (3, 3, 128, 64))
    # CBL13 = con2d(CBL12, (1, 1, 64, 128))
    CBL14 = con2d(CBL12, (3, 3, 64, 128))
    # CBL15 = con2d(CBL14, (1, 1, 64, 128))

    route_1 = CBL14
    CBL16 = con2d(CBL14, (3, 3, 128, 256), downsample=True)
    conv_sbbox = con2d(CBL16, (2, 2, 256, 3 * (NUM_CLASS + 5)), bn=False, downsample=True)

    ZCB1 = con2d(route_1, (3, 3, 128, 256), downsample=True)
    # ZCB1_1 =  Conv2D(256, 2)(ZCB1)

    layer_add3 = tf.concat([ZCB1, route_2], -1)

    CBL17 = con2d(layer_add3, (1, 1, 512, 256))
    CBL18 = con2d(CBL17, (3, 3, 256, 128))
    # CBL19 = con2d(CBL18, (1, 1, 128, 256))
    CBL20 = con2d(CBL18, (3, 3, 128, 256))
    # CBL21 = con2d(CBL20, (1, 1, 128, 256))

    route_2 = CBL20
    CBL22 = con2d(CBL20, (3, 3, 256, 512),downsample=True)
    conv_mbbox = con2d(CBL22, (2, 2, 512, 3 * (NUM_CLASS + 5)), bn=False, downsample=True)

    ZCB2 = con2d(route_2, (3, 3, 256, 512), downsample=True)

    layer_add4 = tf.concat([ZCB2, feature3], -1)

    CBL23 = con2d(layer_add4, (1, 1, 1024, 512))
    CBL24 = con2d(CBL23, (3, 3, 512, 1024))
    CBL25 = con2d(CBL24, (1, 1, 1024, 512))
    CBL26 = con2d(CBL25, (3, 3, 512, 1024))
    CBL27 = con2d(CBL26, (1, 1, 1024, 512))
    CBL28 = con2d(CBL27, (3, 3, 512, 1024), downsample=True)
    conv_lbbox = con2d(CBL28, (2, 2, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False, downsample=True)
    print(conv_sbbox.shape)
    print(conv_mbbox.shape)
    print(conv_lbbox.shape)
    return [conv_sbbox, conv_mbbox, conv_lbbox]

