import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import Mul
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf



# Position encodeing
def get_angles(pos, i, d_model):
    angle_rates =  1/ np.power(10000,(2*(i//2)) / np.float32(d_model))
    return pos*angle_rates


def positional_encoding(position, d_model):
    """
    n, d = 2048, 512
    pos_encoding = positional_encoding(n, d) #(1, position, d_model)
    pos_encoding = pos_encoding[0]
    pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2)) # (position, d_model/2, 2)
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0)) # (2, d_model/2, position)
    pos_encoding = tf.reshape(pos_encoding, (d, n)) #(d_model, position)
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]  #(1, position, d_model)

    return tf.cast(pos_encoding, dtype=tf.float32)

#  decoder padding masking
def creare_padding_mask(seq):
    """
    x = tf.constant([[7,6,0,0,1], [1,2,3,0,0], [0,0,0,4,5]]) \n
    z  = creare_padding_mask(x)\n

    print(z)\n

    tf.Tensor(\n
    [[[[0. 0. 1. 1. 0.]]]\n
     [[[0. 0. 0. 1. 1.]]]\n
     [[[1. 1. 1. 0. 0.]]]], shape=(3, 1, 1, 5), dtype=float32)\n
    """
    seq = tf.cast(tf.math.equal(seq,0), tf.float32)

    # add extra dimnsions to add padding
    # to the attention logits.

    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1 , 1 , seq_len)

def create_look_ahead_mask(size):
    """
    x = tf.random.uniform((1,3))\n
    temp = create_look_ahead_mask(x.shape[1])\n
    print(temp)\n
    tf.Tensor(\n
    [[0. 1. 1.]\n
     [0. 0. 1.]\n
     [0. 0. 0.]], shape=(3, 3), dtype=float32)
    """
    mask = 1 - tf.linalg.band_part(tf.ones([size, size]), -1, 0)
    return mask # (seq_len, seq_len)

def scaled_dot_product_attention(q,k,v,mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v muist have matching penultimate dimension, i.e. : seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable of addition.

    ex. Decoder masking

    Args:
        q: query shape = (..., seq_len_q, depth)
        k: key shape = (..., seq_len_k, depth)
        v: value shape = (..., seq_len_v, depth)
        mask : float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b= True) # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask* -1e9)

    attention_weigths = tf.nn.softmax(scaled_attention_logits, axis=-1) #(..., seq_len_q, seq_len_k)
    output =  tf.matmul(attention_weigths, v)                           #(..., seq_len_q, depth_v)
    return output, attention_weigths

def print_out(q, k, v):
    """
    output test for scaled_dot_product_attention .

    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10, 0, 0],
                        [0, 10, 0],
                        [0, 0, 10],
                        [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                        [10, 0],
                        [100, 5],
                        [1000, 6]], dtype=tf.float32)  # (4, 2)

    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    print_out(temp_q, temp_k, temp_v)
    """
    temp_out, temp_attn = scaled_dot_product_attention(
    q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    temp_mha = MultiHeadAttention(d_model=512, num_heads = 8) \n
    y = tf.random.uniform((1, 60, 512)) \n
    out, attn = temp_mha(v=y, k=y, q=y, mask=None) \n
    print(out.shape, attn.shape) \n
    (1, 60, 512) (1, 8, 60, 60)

    input, output shape :  (batch_size, encoder_sequence, d_model) \n
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0 

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x,  batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads. seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights =  scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])   # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, 
                               (batch_size, -1, self.d_model))              # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)                               # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    sample_ffn = point_wise_feed_forward_network(512, 2048) \n
    sample_ffn(tf.random.uniform((64,50,512))).shape \n
    TensorShape([64, 50, 512]) \n
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation= "relu"),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """Each encoder layer consists of a lower layer.
    Multi-headed attention (with padded mask) Point wise feed forward network.

    sample_encoder_layer = EncoderLayer(512, 8, 2048).
    sample_encoder_layer_output =  sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None).
    pritn(sample_encoder_layer_output.shape).
    (batch_size, input_seq_len, d_model).
    (64,43,512)
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)                                    #(batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)                                     #(batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)                                                 #(batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training = training)                 #(batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)                                   #(batch_size, input_seq_len, d_model)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    """
    Each decoder layer consists of a sublayer. \n
    Masked multi-head attractions (including preview masks and padding masks) \n
    The multi-header (including padding masks) \n
    V (value) and K (key) receive the encoder output as input \n
    Q (queries) receive the output from the sub-layer of the masked multi-header. \n
    Point wise feed forward network. \n

    sample_decoder_layer = DecoderLayer(512, 8, 2048) \n
    sample_decoder_layer_output, _, _ = sample_decoder_layer( \n
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, \n
        False, None, None) \n
    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model) \n
    (64,43,512)
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1  = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2  = self.mha2(enc_output, enc_output, out1, padding_mask)
        # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layernorm2(attn2 + out1)
        # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training = training)
        out3 = self.layernorm3(ffn_output + out2)
        # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    """Encoder consists of : Input Embedding, Positional Encoding, N Encoder Layer
    Inputs are made through embeddings that are summed by Positional encoding.
    The output of this sum is input to the encoder layer.
    The output of the encoder is input to the decoder.

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                enc_output=sample_encoder_output,
                                training=False,
                                look_ahead_mask=None,
                                padding_mask=None)

    output.shape, attn['decoder_layer2_block2'].shape

    (TensorShape([64, 26, 512]), TensorShape([64, 8, 26, 62]))
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.

        x = self.embedding(x)       # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x= self.enc_layers[i](x, training, mask)
        
        return x                    # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    """Decoder consists of: Output Embedding, Position Encoding, N Decoder Layer
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                enc_output=sample_encoder_output,
                                training=False,
                                look_ahead_mask=None,
                                padding_mask=None)
    print(output.shape, attn['decoder_layer2_block2'].shape)
    TensorShape([64, 26, 512]), TensorShape([64, 8, 26, 62])
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)           # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 =  self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
        
class Transformer(tf.keras.Model):
    """
    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                enc_padding_mask=None,
                                look_ahead_mask=None,
                                dec_padding_mask=None)

    fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
    TensorShape([64, 36, 8000])
    
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                                input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                                target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
            look_ahead_mask, dec_padding_mask):
        
        enc_output = self.tokenizer(inp, training, enc_padding_mask)        #(batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weigths = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)                         #(batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weigths