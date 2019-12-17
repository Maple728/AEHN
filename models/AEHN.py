#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/21 21:49
@desc:
"""
import tensorflow as tf
from tensorflow.keras import layers

from models.base_model import BaseModel


class Attention:
    def __init__(self, cell_units, reuse=tf.AUTO_REUSE):
        self.cell_units = cell_units
        with tf.variable_scope('attention_layer', reuse=reuse):
            self.attention_w1 = layers.Dense(self.cell_units, name='w1')
            self.attention_w2 = layers.Dense(self.cell_units, name='w2')
            self.attention_v = layers.Dense(1, name='v')

    def compute_attention_weight(self, decoder_state, encoder_output, pos_mask=None):
        """
        :param decoder_state: (batch_size, num_queries, cell_units)
        :param encoder_output: (batch_size, num_keys, cell_units)
        :param pos_mask: ['self-right', 'right', None]
        :return: (batch_size, num_queries, cell_units), (batch_size, num_queries, num_keys, 1)
        """
        MASKED_VAL = - 2 ** 32 + 1
        # (batch_size, num_queries, 1, cell_units)
        q = decoder_state[:, :, None, :]
        # (batch_size, 1, num_keys, cell_units)
        k = encoder_output[:, None, :, :]

        # (batch_size, num_queries, num_keys, cell_units)
        weighted_state = self.attention_w1(q) + self.attention_w2(k)

        # (batch_size, num_queries, num_keys, 1)
        score = self.attention_v(tf.nn.tanh(weighted_state)) / tf.sqrt(tf.cast(self.cell_units, dtype=tf.float32))

        if pos_mask:
            # (batch_size, num_queries, num_keys)
            score = tf.squeeze(score, axis=-1)

            ones_mat = tf.ones_like(score)
            masked_val_mat = ones_mat * MASKED_VAL

            # (batch_size, num_queries, num_keys)
            masks = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            if pos_mask == 'self-right':
                # (batch_size, num_queries, num_keys)
                score = tf.where(tf.equal(masks, 0),
                                 masked_val_mat,
                                 score)
            elif pos_mask == 'right':
                # transpose to upper triangle
                pos_mask = tf.transpose(masks, perm=[0, 2, 1])
                score = tf.where(tf.equal(pos_mask, 0),
                                 score,
                                 masked_val_mat)
            else:
                raise RuntimeError('Unknown pas_mask: {}'.format(pos_mask))

            # (batch_size, num_queries, num_keys, 1)
            score = tf.expand_dims(score, axis=-1)

        # (batch_size, num_queries, num_keys, 1)
        attention_weight = tf.nn.softmax(score, axis=-2)

        # (batch_size, num_queries, num_keys, cell_units)
        context_vector = attention_weight * k

        # (batch_size, num_queries, cell_units)
        context_vector = tf.reduce_sum(context_vector, axis=-2)

        return context_vector, attention_weight


class AEHN(BaseModel):
    """
    AEHN with various length of input.
    """

    def train(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        lr = kwargs.get('lr')
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.learning_rate: lr
        }
        _, loss, pred_types, pred_time = sess.run([self.train_op, self.loss, self.pred_types, self.pred_time],
                                                  feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:]
        }
        return loss, preds, labels

    def predict(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs
        }
        loss, pred_types, pred_time = sess.run([self.loss, self.pred_types, self.pred_time],
                                               feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:]
        }
        return loss, preds, labels

    def __init__(self, model_config):
        # get hyperparameters from config
        self.process_dim = model_config['process_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_integral_sample = model_config.get('num_integral_sample', 100)
        self.max_time_integral = model_config.get('max_time_integral', 2.0)

        with tf.variable_scope('AEHN'):
            # --------------- placeholders -----------------
            # train placeholder
            self.learning_rate = tf.placeholder(tf.float32)
            # input placeholder
            # shape -> [batch_size, max_len]
            self.types_seq = tf.placeholder(tf.int32, shape=[None, None])
            self.dtimes_seq = tf.placeholder(tf.float32, shape=[None, None])

            # --------------- build model -----------------
            # shape -> [batch_size, max_len, process_dim]
            self.types_seq_one_hot = tf.one_hot(self.types_seq, self.process_dim)
            # 1. Embedding of input
            # shape -> [batch_size, max_len, hidden_dim]
            type_seq_emb = self.embedding_layer(self.types_seq)

            # 2. Intensity layer
            # shape -> [batch_size, max_len, hidden_size]
            lambdas = self.intensity_layer(type_seq_emb)

            # 3. Inference layer
            # [batch_size, max_len, process_dim], [batch_size, max_len]
            pred_type_logits, pred_time = self.inference_layer(lambdas)

            # 4. train step
            self.loss = self.compute_all_loss(pred_type_logits, pred_time)
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.opt.minimize(self.loss)

            # assign prediction
            # shape -> [batch_size, max_len]
            self.pred_types = tf.argmax(pred_type_logits, axis=-1)
            # shape -> [batch_size, max_len]
            self.pred_time = pred_time

    def embedding_layer(self, x_input):
        # add 1 dim because of EOS padding
        emb_layer = layers.Embedding(self.process_dim + 1, self.hidden_dim, name='type_embedding')
        # shape -> [batch_size, max_len, hidden_dim]
        emb = emb_layer(x_input)
        return emb

    def intensity_layer(self, x_input, reuse=tf.AUTO_REUSE):
        """
        :param x_input: [batch_size, max_len, hidden_dim]
        :param reuse:
        :return: [batch_size, max_len, hidden_dim]
        """
        with tf.variable_scope('intensity_layer', reuse=reuse):
            delta_layer = layers.Dense(self.hidden_dim, activation=tf.nn.softplus, name='delta_layer')
            attention_layer = Attention(self.hidden_dim)
            mu_layer = layers.Dense(self.hidden_dim, activation=tf.nn.softplus, name='mu_layer')

        with tf.name_scope('intensity_layer'):
            max_len = tf.shape(x_input)[1]
            # compute mu
            # shape -> [batch_size, max_len, hidden_dim]
            mus = mu_layer(x_input)

            # compute alpha
            # (batch_size, max_len, max_len, 1)
            _, all_attention_weights = attention_layer.compute_attention_weight(x_input,
                                                                                x_input,
                                                                                pos_mask='right')
            # shape -> [batch_size, max_len, max_len, 1]
            alphas = all_attention_weights

            # compute delta
            # shape -> [batch_size, max_len, max_len, hidden_dim]
            left = tf.tile(x_input[:, None, :, :], [1, max_len, 1, 1])
            right = tf.tile(x_input[:, :, None, :], [1, 1, max_len, 1])
            # shape -> [batch_size, max_len, max_len, hidden_dim * 2]
            cur_prev_concat = tf.concat([left, right], axis=-1)
            # shape -> [batch_size, max_len, max_len, hidden_dim]
            deltas = delta_layer(cur_prev_concat)

            # compute time elapse
            # shape -> [batch_size, max_len]
            # [dt_0, dt_1, dt_2] => [dt_1 + dt_2, dt_2, 0]
            cum_dtimes = tf.cumsum(self.dtimes_seq, axis=1, reverse=True, exclusive=True)
            # shape -> [batch_size, max_len, max_len, 1] (positive)
            elapses = tf.expand_dims(cum_dtimes[:, None, :] - cum_dtimes[:, :, None], axis=-1)

            # compute lambda (mu + sum<alpha * exp(-delta * elapse)>)
            # shape -> [batch_size, max_len, hidden_dim]
            left_term = mus
            # shape -> [batch_size, max_len, hidden_dim]
            right_term = tf.reduce_sum(alphas * tf.exp(-deltas * elapses), axis=-2)
            # shape -> [batch_size, max_len, hidden_dim]
            lambdas = left_term + right_term

            return lambdas

    def inference_layer(self, lambdas):
        """
        :param lambdas: [batch_size, max_len, hidden_dim]
        :return:
        """
        type_inference_layer = layers.Dense(self.process_dim, activation=None)
        time_inference_layer = layers.Dense(1, activation=None)

        # shape -> [batch_size, max_len, process_dim]
        pred_type_logits = type_inference_layer(lambdas)
        # shape -> [batch_size, max_len]
        pred_time = tf.squeeze(time_inference_layer(lambdas), axis=-1)

        return pred_type_logits, pred_time

    def compute_all_loss(self, pred_type_logits, pred_times):
        with tf.variable_scope('loss'):
            # EOS padding type is all zeros in the last dim of the tensor
            seq_mask = tf.reduce_sum(self.types_seq_one_hot[:, 1:], axis=-1) > 0

            # (batch_size, max_len - 1, process_dim)
            pred_type_logits = pred_type_logits[:, :-1]

            pred_type_logits = pred_type_logits - tf.reduce_max(pred_type_logits, axis=-1, keepdims=True)

            type_label = self.types_seq_one_hot[:, 1:]

            pred_type_proba = tf.nn.softmax(pred_type_logits, axis=-1) + 1e-31

            # (batch_size, max_len - 1)
            cross_entropy = tf.reduce_sum(- tf.log(pred_type_proba) * type_label, axis=-1)

            type_loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, seq_mask))

            dtimes_pred = pred_times[:, :-1]
            dtimes_label = self.dtimes_seq[:, 1:]

            time_diff = tf.boolean_mask(dtimes_pred - dtimes_label, seq_mask)

            time_loss = tf.reduce_mean(tf.abs(time_diff))

            return type_loss
