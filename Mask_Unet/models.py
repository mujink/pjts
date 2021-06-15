from tensorflow.python.keras.layers.convolutional import ZeroPadding2D, Conv2D
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.python.ops.gen_math_ops import sigmoid
from tensorflow.keras.activations import sigmoid
from tensorflow.python.ops.gen_nn_ops import conv2d
from model_block import con2d, con2dTrans, res_block, upsample
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def initlayer(input_layer):
# def initlayer(input_layer, NUM_CLASS):
    
    CBL1 = con2d(input_layer, (3, 3,  3,  32))
    resloop = con2d(CBL1, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        resloop = res_block(resloop,  64,  32, 64)

    L = resloop

    resloop = con2d(resloop, (3, 3,  64, 128), downsample=True)
    for i in range(2):
        resloop = res_block(resloop, 128,  64, 128)

    M =resloop

    resloop = con2d(resloop, (3, 3, 128, 256), downsample=True)

    for i in range(4):
        resloop = res_block(resloop, 256, 128, 256)

    # resloop = con2d(resloop, (1, 1, 128, 3 * (NUM_CLASS + 5)), bn=False, downsample=True)

    S = resloop

    return S, M, L


def Unet(input_layer):

    S, M, L = initlayer(input_layer)

    resloop = S
    for i in range(4):
        resloop = res_block(resloop, 256, 128, 256)

    CBL2 = con2dTrans(resloop, (3, 3, 256, 128), up_sample=True)
    # layer_add1 = tf.concat([CBL2, M], -1)
    layer_add1 = CBL2 + M

    for i in range(2):
        resloop = res_block(layer_add1, 128,  64, 128)

    CBL3 = con2dTrans(resloop, (3, 3, 128, 64), up_sample=True)

    # layer_add2 = tf.concat([CBL3, L], -1)
    layer_add2 = CBL3 + L

    for i in range(1):
        resloop = res_block(layer_add2, 64,  32, 64)

    resloop = con2dTrans(resloop, (3, 3, 64, 32), up_sample=True)

    out = Conv2D(filters=3, kernel_size=(3,3),strides=(1, 1), padding="same",name="output")(resloop)
    out = BatchNormalization()(out)
    out = sigmoid(out)
    return out

# inputs = Input(shape=(416,416,3))
# out = Unet(inputs)
# # s,m,l = initlayer(inputs)
# model = Model(inputs=inputs, outputs=out)
# model.summary()