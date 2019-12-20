#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/21 21:49
@desc:
"""
from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import layers


class BaseModel(object):

    def __init__(self, model_config):
        # get hyperparameters from config
        self.process_dim = model_config['process_dim']
        self.hidden_dim = model_config['hidden_dim']

        self.n_loss_integral_sample = model_config.get('n_loss_integral_sample', 10)
        self.n_pred_integral_sample = model_config.get('n_pred_integral_sample', 50)
        self.max_time_pred = model_config.get('max_time_pred', 1.0)

        self.pred_method = model_config.get('pred_method')

        with tf.variable_scope('model_input'):
            # --------------- placeholders -----------------
            # train placeholder
            self.learning_rate = tf.placeholder(tf.float32)
            # input placeholder
            # shape -> [batch_size, max_len]
            self.types_seq = tf.placeholder(tf.int32, shape=[None, None])
            # dt_i = t_i - t_{i-1}
            self.dtimes_seq = tf.placeholder(tf.float32, shape=[None, None])

            # --------------- input process -----------------
            # shape -> [batch_size, max_len, process_dim]
            self.types_seq_one_hot = tf.one_hot(self.types_seq, self.process_dim)
            # EOS padding type is all zeros in the last dim of the tensor
            self.seq_mask = tf.reduce_sum(self.types_seq_one_hot, axis=-1) > 0

    @abstractmethod
    def train(self, sess, batch_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, sess, batch_data, **kwargs):
        pass

    @staticmethod
    def generate_model_from_config(model_config):
        model_name = model_config.get('name')

        for subclass in BaseModel.__subclasses__():
            if subclass.__name__ == model_name:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_name)

    # ------------------ general functions for subclasses --------------------
    def hybrid_inference(self, lambdas):
        process_dim = self.process_dim
        with tf.variable_scope('hybrid_inference'):
            # create layers
            type_inference_layer = layers.Dense(process_dim, activation=None)
            time_inference_layer = layers.Dense(1, activation=None)

            # computation
            # shape -> [batch_size, max_len, process_dim]
            pred_type_logits = type_inference_layer(lambdas)
            # shape -> [batch_size, max_len]
            pred_time = tf.squeeze(time_inference_layer(lambdas), axis=-1)

        return pred_type_logits, pred_time

    def loglikelihood_inference(self, lambdas_pred_samples, dtimes_pred_samples):
        """ Predict the type and time from intensity function over infinite time.
        :param lambdas_pred_samples: [batch_size, max_len, n_pred_sample, process_dim]
        :param dtimes_pred_samples: [batch_size, max_len, n_pred_sample]
        :return:
        """
        with tf.variable_scope('intensity_inference'):
            # compute density
            # [batch_size, max_len, n_pred_sample]
            lambdas_total_samples = tf.reduce_sum(lambdas_pred_samples, axis=-1)
            # [batch_size, max_len, n_pred_sample]
            integral_samples = self.get_acc_rect_integral(lambdas_total_samples,
                                                          self.max_time_pred / self.n_pred_integral_sample)
            # [batch_size, max_len, n_pred_sample]
            density_samples = lambdas_total_samples * tf.exp(-integral_samples)

            # compute time prediction
            # [batch_size, max_len]
            pred_times = self.trapezium_integral(density_samples * dtimes_pred_samples, dtimes_pred_samples)

            # compute type prediction
            # [batch_size, max_len, n_pred_sample, process_dim]
            type_ratio_samples = lambdas_pred_samples / lambdas_total_samples[:, :, :, None]

            # [batch_size, max_len, process_dim]
            pred_type_logits = self.trapezium_integral(type_ratio_samples * density_samples[:, :, :, None],
                                                       dtimes_pred_samples[:, :, :, None])

        return pred_type_logits, pred_times

    def shuffle_hybrid_loss(self, pred_type_logits, pred_times):
        """ Calculate the total loss of all points in the sequence by using the latter point as the label predicted by
        the previous point.
        :param pred_type_logits: [batch_size, max_len, process_dim]
        :param pred_times: [batch_size, max_len]
        :return:
        """
        label_types = self.types_seq
        label_times = self.dtimes_seq
        seq_mask = self.seq_mask
        with tf.variable_scope('shuffle_hybrid_loss'):
            # shape -> [batch_size, max_len - 1]
            seq_mask = seq_mask[:, 1:]
            # (batch_size, max_len - 1, process_dim)
            type_label = label_types[:, 1:]

            pred_type_logits = pred_type_logits[:, :-1]
            pred_type_logits = pred_type_logits - tf.reduce_max(pred_type_logits, axis=-1, keepdims=True)
            pred_type_proba = tf.nn.softmax(pred_type_logits, axis=-1) + 1e-8

            # (batch_size, max_len - 1)
            cross_entropy = tf.reduce_sum(- tf.log(pred_type_proba) * type_label, axis=-1)
            type_loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, seq_mask))

            dtimes_pred = pred_times[:, :-1]
            dtimes_label = label_times[:, 1:]

            time_diff = tf.boolean_mask(dtimes_pred - dtimes_label, seq_mask)
            time_loss = tf.reduce_mean(tf.abs(time_diff))

            return type_loss + time_loss

    def shuffle_loglikelihood_loss(self, lambdas, lambdas_loss_samples, dtimes_loss_samples):
        """

        :param lambdas: [batch_size, max_len, process_dim]
        :param lambdas_loss_samples: [batch_size, max_len, n_loss_sample, process_dim]
        :param dtimes_loss_samples: [batch_size, max_len, n_loss_sample]
        :return:
        """
        with tf.variable_scope('shuffle_loglikelihood_loss'):
            # Shuffle label and prediction

            # label move forward one point
            # [batch_size, max_len - 1, process_dim]
            seq_event_onehot = self.types_seq_one_hot[:, 1:]
            # [batch_size, max_len - 1]
            seq_zero_mask = tf.reduce_sum(seq_event_onehot, axis=-1)
            # [batch_size, max_len - 1, n_loss_sample]
            dtimes_loss_samples = dtimes_loss_samples[:, :-1]

            # prediction truncate the last point (no label)
            # [batch_size, max_len - 1, process_dim]
            lambdas = lambdas[:, :-1]
            # [batch_size, max_len - 1, n_loss_sample]
            lambdas_total_samples = tf.reduce_sum(lambdas_loss_samples[:, :-1], axis=-1)

            # Compute loglikelihood loss

            # shape -> [batch_size, max_len - 1]
            target_lambdas = tf.reduce_sum(lambdas * seq_event_onehot, axis=-1)
            target_lambdas_masked = tf.boolean_mask(target_lambdas, target_lambdas > 0)
            term_1 = tf.reduce_sum(tf.log(target_lambdas_masked))

            # shape -> [batch_size, max_len - 1]
            lambdas_integral = self.trapezium_integral(lambdas_total_samples, dtimes_loss_samples)
            term_2 = tf.reduce_sum(seq_zero_mask * lambdas_integral)

            events_loss = - (term_1 - term_2)
            n_event = tf.reduce_sum(seq_zero_mask)

            return events_loss / n_event

    def get_acc_rect_integral(self, values, interval):
        """ Calculate the accumulated rectangular integral over dim-1 (samples dim)
        :param values: [batch_size, None, n_samples, ...]
        :param interval: float type
        :return: the shape is same sa values
        """
        integral = tf.cumsum(values * interval, axis=2)
        return integral

    def trapezium_integral(self, values, dtimes):
        """ Trapezium integral over dim-1 (samples dim).
        :param values: [batch_size, None, n_samples, ...]
        :param dtimes: the shape is same as values  or float
        :return: [batch_size, None, ...]
        """
        if isinstance(dtimes, float):
            heights = dtimes
        else:
            heights = dtimes[:, :, 1:] - dtimes[:, :, :-1]
        upper_bottom = values[:, :, 1:] + values[:, :, :-1]
        integral = tf.reduce_sum(0.5 * heights * upper_bottom, axis=2)
        return integral


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
            zeros_mat = tf.zeros_like(score)
            masked_val_mat = ones_mat * MASKED_VAL

            # (batch_size, num_queries, num_keys)
            lower_diag_masks = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            if pos_mask == 'self-right':
                # (batch_size, num_queries, num_keys)
                score = tf.where(tf.equal(lower_diag_masks, 0),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_diag_masks, 0),
                                            zeros_mat,
                                            attention_weight)
            elif pos_mask == 'right':
                # transpose to upper triangle
                lower_masks = tf.transpose(lower_diag_masks, perm=[0, 2, 1])

                score = tf.where(tf.equal(lower_masks, 1),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_masks, 1),
                                            zeros_mat,
                                            attention_weight)

            else:
                raise RuntimeError('Unknown pas_mask: {}'.format(pos_mask))

            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.expand_dims(attention_weight, axis=-1)
        else:

            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.nn.softmax(score, axis=-2)

        # (batch_size, num_queries, num_keys, cell_units)
        context_vector = attention_weight * k

        # (batch_size, num_queries, cell_units)
        context_vector = tf.reduce_sum(context_vector, axis=-2)

        return context_vector, attention_weight
