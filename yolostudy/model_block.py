from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D
import tensorflow as tf


"""
==========================================================================================
                CM      CL      CBL     CBM     ZCL     ZCM     ZCBL    ZCBM
downsample      X       X       X       X       O       O       O       O
activate        M       L       L       M       L       M       L       M
bn              X       X       O       O       X       X       O       O

activate(Default)   : leak_relu
downsample(Default) : False
bn(Default)         : True
layer(Default)      : CBL
==========================================================================================
CM  : Conv2D + LeaK_relu 
CL  : Conv2D + mish
CBL : Conv2D +  batchnormal + leak_relu
CBM : Conv2D +  batchnormal + mish
ZCL : ZeroPadding2D + CL
ZCM : ZeroPadding2D + CM
ZCBL : ZeroPadding2D + CBL
ZCBM : ZeroPadding2D + CBM
"""

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def con2d(input_layer, filters_shape, downsample=False, activate="l", bn=True):
    """
    =================================================
    filters_shape : len > 2
    init:
        (kernel_x, kernely, in channel, out channel)

    conv filtter,  kernel init : 
        filters_shape[0]    =   kernel_size
        filters_shape[-1]   =   fliters

    strides :
        downsample True  => 2
        downsample False(Default) => 1
    pading  :
        downsample True   => valid
        downsample False(Default)  => same
    =================================================
    """
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)

    if not activate == None :
        if activate == "l":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate == "m":
            conv = mish(conv)
    return conv


def res_block(input_layer, input_channel, filter_num1, filter_num2, activate='l'):
    """
    =================================================
    ResNet Block
    init
    input_layer =>  CBL => CBL => feature map
    input_layer = short_cut
    out = short_cut + conv
    =================================================
    """
    short_cut = input_layer
    conv = con2d(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate=activate)
    conv = con2d(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate=activate)

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
