import tensorflow as tf

def decode_train(conv_output, height_size, width_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):


    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], width_size, height_size, 3, 5 + NUM_CLASS))
    # conv_output
    # (None, 212, 308, 3, 2373)
    # (None, 106, 154, 3, 2373)
    # (None, 53, 77, 3, 2373)
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),-1)
    # (None, 212, 308, 3, 2)
    # (None, 212, 308, 3, 2)
    # (None, 212, 308, 3, 1)
    # (None, 212, 308, 3, 2368)
    # (None, 106, 154, 3, 2)
    # (None, 106, 154, 3, 2)
    # (None, 106, 154, 3, 1)
    # (None, 106, 154, 3, 2368)
    # (None, 53, 77, 3, 2)
    # (None, 53, 77, 3, 2)
    # (None, 53, 77, 3, 1)
    # (None, 53, 77, 3, 2368)
    xy_grid = tf.meshgrid(tf.range(height_size), tf.range(width_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    # xy_grid
    # (None, 212, 308, 3, 2)
    # (None, 106, 154, 3, 2)
    # (None, 53, 77, 3, 2)
    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], -1)
    # (None, 212, 308, 3, 4)
    # (None, 106, 154, 3, 4)
    # (None, 53, 77, 3, 4)
    pred_conf = tf.sigmoid(conv_raw_conf)
    # (None, 212, 308, 3, 1)
    # (None, 106, 154, 3, 1)
    # (None, 53, 77, 3, 1)
    pred_prob = tf.sigmoid(conv_raw_prob)
    # (None, 212, 308, 3, 2368)
    # (None, 106, 154, 3, 2368)
    # (None, 53, 77, 3, 2368)
    return tf.concat([pred_xywh, pred_conf, pred_prob], -1)