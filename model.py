#! -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

# macOS style
from keras.layers import TimeDistributed, Dense, LayerNormalization, MultiHeadAttention
# other platforms
# from tensorflow.python.keras.layers import TimeDistributed, Dense

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Add, Conv1D, Lambda, Dropout, Flatten, Activation
from tensorflow.python.keras.layers import Input, Embedding
from tensorflow.python.keras import backend as K


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def get_sub_mask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


# class ScaledDotProductAttention:
#     def __init__(self, d_model, attn_dropout=0.1):
#         self.temper = np.sqrt(d_model)
#         self.dropout = Dropout(attn_dropout)
#
#     def __call__(self, q, k, v, mask):
#         attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
#         if mask is not None:
#             _mask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
#             attn = Add()([attn, _mask])
#         attn = Activation('softmax')(attn)
#         attn = self.dropout(attn)
#         output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
#         return output, attn


# class MultiHeadAttention:
#     def __init__(self, n_head, d_model, d_k, d_v, dropout, use_norm=True):
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#         self.dropout = dropout
#         self.qs_layer = Dense(n_head * d_k, use_bias=False)
#         self.ks_layer = Dense(n_head * d_k, use_bias=False)
#         self.vs_layer = Dense(n_head * d_v, use_bias=False)
#         self.attention = ScaledDotProductAttention(d_model)
#         self.layer_norm = LayerNormalization() if use_norm else None
#         self.w_o = TimeDistributed(Dense(d_model))
#
#     def __call__(self, q, k, v, mask=None):
#         d_k, d_v = self.d_k, self.d_v
#         n_head = self.n_head
#         head = None
#         attn = None
#
#         qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
#         ks = self.ks_layer(k)
#         vs = self.vs_layer(v)
#
#         def reshape1(x):
#             s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
#             x = tf.reshape(x, [s[0], s[1], n_head, d_k])
#             x = tf.transpose(x, [2, 0, 1, 3])
#             x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
#             return x
#
#         qs = Lambda(reshape1)(qs)
#         ks = Lambda(reshape1)(ks)
#         vs = Lambda(reshape1)(vs)
#
#         if mask is not None:
#             mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
#         head, attn = self.attention(qs, ks, vs, mask=mask)
#
#         def reshape2(x):
#             s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
#             x = tf.reshape(x, [n_head, -1, s[1], s[2]])
#             x = tf.transpose(x, [1, 2, 0, 3])
#             x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
#             return x
#
#         head = Lambda(reshape2)(head)
#
#         outputs = self.w_o(head)
#         outputs = Dropout(self.dropout)(outputs)
#         if not self.layer_norm:
#             return outputs, attn
#         outputs = Add()([outputs, q])
#         return self.layer_norm(outputs), attn


class PositionwiseFeedForward:
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        # output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.self_att_layer(enc_input, enc_input, enc_input, attention_mask=mask)
        output = self.pos_ffn_layer(output)
        return output


class Encoder:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=2, dropout=0.1):
        self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, x, mask=None, active_layers=999):
        output = self.emb_dropout(x)
        atts = None
        for enc_layer in self.layers[:active_layers]:
            output = enc_layer(output, mask)
        return output


class PRM:
    def __init__(self, seq_len, d_feature, d_model=64, d_inner_hid=128, n_head=1, d_k=64, d_v=64, layers=2,
                 dropout=0.1):
        self.model = None
        self.seq_len = seq_len
        self.d_feature = d_feature
        self.d_model = d_model
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout)

    def build(self, pos_mode=0, use_mask=False, active_layers=999):
        v_input = Input(shape=(self.seq_len, self.d_feature), name='v_input')
        d0 = TimeDistributed(Dense(self.d_model))(v_input)
        pos_input = Input(shape=(self.seq_len,), dtype='int32', name='pos_input')
        if pos_mode == 0:  # no pos embedding
            p0 = None
        elif pos_mode == 1:  # use trainable pos embedding
            pos_embedding = Embedding(self.seq_len, self.d_model)
            p0 = pos_embedding(pos_input)
        if p0 is not None:
            combine_input = Add()([d0, p0])
        else:
            combine_input = d0  # no pos
        sub_mask = None
        if use_mask:
            sub_mask = Lambda(get_sub_mask)(pos_input)
        enc_output = self.encoder(combine_input, mask=sub_mask, active_layers=active_layers)
        # score
        time_score_dense1 = TimeDistributed(Dense(self.d_model, activation='tanh'))(enc_output)
        time_score_dense2 = TimeDistributed(Dense(1))(time_score_dense1)
        flat = Flatten()(time_score_dense2)
        score_output = Activation(activation='softmax')(flat)
        self.model = Model([pos_input, v_input], score_output)
        return self.model
