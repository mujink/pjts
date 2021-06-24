from tensorflow.python.keras.layers.convolutional import ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.layers import Bidirectional, Reshape, LSTM, GRU, Dense, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization, Dense
# from tensorflow.python.ops.gen_math_ops import sigmoid
from tensorflow.keras.activations import sigmoid, tanh, softmax, linear
from tensorflow.python.ops.gen_nn_ops import conv2d
from model_block import con2d, con2dTrans, res_block, upsample
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def initlayer(input_layer):
# def initlayer(input_layer, NUM_CLASS):
    
    CBL1 = con2d(input_layer, (3, 3,  3,  32))
    resloop = con2d(CBL1, (3, 3, 32,  64), downsample=True)
    # resloop = AveragePooling2D(pool_size=(2, 2),padding="same")(resloop)
    for i in range(1):
        resloop = res_block(resloop,  64,  32, 64)

    L = resloop

    resloop = con2d(resloop, (3, 3,  64, 128), downsample=True)
    # resloop = AveragePooling2D(pool_size=(2, 2),padding="same")(resloop)

    for i in range(2):
        resloop = res_block(resloop, 128,  64, 128)

    M =resloop

    resloop = con2d(resloop, (3, 3, 128, 256), downsample=True)
    # resloop = AveragePooling2D(pool_size=(2, 2),padding="same")(resloop)

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

    out = Conv2D(filters=1, kernel_size=(3,3),strides=(1, 1), padding="same",name="output")(resloop)
    out = BatchNormalization()(out)
    out = sigmoid(out)
    # out = softmax(out)
    # out = Dense((input_layer*input_layer*1))(out)
    return out
    out = tanh(out)

# inputs = Input(shape=(416,416,3))
# out = Unet(inputs)
# # s,m,l = initlayer(inputs)
# model = Model(inputs=inputs, outputs=out)
# model.summary()

def CRNN(input, num_classes, gru=False):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
    """
    
    act = 'relu'
    
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(input)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)
    
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)
    
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='batchnorm1')(x)
    
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    x = BatchNormalization(name='batchnorm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='pool5', padding='valid')(x)
    
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)
    x = Reshape((-1,512))(x)
    
    if gru:
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Bidirectional(GRU(256, return_sequences=True))(x)
    
    else:
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    x = Dense(num_classes, name='dense1')(x)
    
    # output은 softmax함수를 사용하여 라벨에대한 확률값이 나온다.
    y_pred = Activation('softmax', name='softmax')(x)
    
    return y_pred

from tensorflow.keras import layers

def get_model(inputs, num_classes=3):

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    return outputs