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

from aehn.models import BaseModel
from aehn.lib.tf_utils import get_variable_weights, get_variable_bias, tensordot
from aehn.lib import tensordot, swap_axes, create_tensor, Attention


class AEHN_mark(BaseModel):
    """
    AEHN with marks.
    """

    def train(self, sess, batch_data, **kwargs):
        type_seqs = batch_data['types']
        dtime_seqs = batch_data['dtimes']
        mark_seqs = batch_data['marks']
        lr = kwargs.get('lr')
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: mark_seqs,
            self.learning_rate: lr
        }

        _, loss, pred_types, pred_time = sess.run([self.train_op, self.loss, self.pred_types, self.pred_time],
                                                  feed_dict=fd)

        # density = sess.run([self.density], feed_dict=fd)
        # print(density)
        # means, stds = sess.run([self.means, self.stds], feed_dict=fd)
        # print(means, stds)

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
        mark_seqs = batch_data['marks']
        fd = {
            self.types_seq: type_seqs,
            self.dtimes_seq: dtime_seqs,
            self.marks_seq: mark_seqs
        }
        loss, pred_types, pred_time, pred_marks = sess.run(
            [self.loss, self.pred_types, self.pred_time, self.pred_marks],
            feed_dict=fd)

        # shape -> [batch_size, max_len - 1]
        preds = {
            'types': pred_types[:, :-1],
            'dtimes': pred_time[:, :-1],
            'marks': pred_marks[:, :-1]
        }
        labels = {
            'types': type_seqs[:, 1:],
            'dtimes': dtime_seqs[:, 1:],
            'marks': mark_seqs[:, 1:]
        }
        return loss, preds, labels

    def __init__(self, model_config):
        # get hyperparameters from config
        BaseModel.__init__(self, model_config)
        with tf.variable_scope('model_input'):
            mark_dim = 1
            # [batch_size, max_len, mark_dim]
            self.marks_seq = tf.placeholder(tf.float32, [None, None, mark_dim])

        with tf.variable_scope('AEHN'):
            # 1. Embedding of input
            # shape -> [batch_size, max_len, hidden_dim]
            type_seq_emb = self.embedding_layer(self.types_seq)
            mark_emb = layers.Dense(self.hidden_dim, activation=tf.nn.relu, name='mark_emb_layer')(self.marks_seq)
            emb = type_seq_emb + mark_emb
            # emb = tf.concat([type_seq_emb, self.marks_seq], axis=-1)
            # Flow layer
            flow_output = self.flow_layer(emb)

            # 2. Intensity layer
            # shape -> [batch_size, max_len, hidden_dim], [batch_size, max_len, n_sample, hidden_dim]
            latent_lambdas, \
            latent_lambdas_loss_samples, dtimes_loss_samples, \
            latent_lambdas_pred_samples, dtimes_pred_samples = self.intensity_layer(flow_output)

            # convert latent lambda to lambda
            lambdas = self.compute_lambda(latent_lambdas)
            lambdas_loss_samples = self.compute_lambda(latent_lambdas_loss_samples)
            lambdas_pred_samples = self.compute_lambda(latent_lambdas_pred_samples)

            # 3. Inference layer and loss function
            # [batch_size, max_len, process_dim], [batch_size, max_len]
            if self.pred_method == 'loglikelihood':
                pred_type_logits, pred_time = self.loglikelihood_inference(lambdas_pred_samples, dtimes_pred_samples)
                self.loss = self.shuffle_loglikelihood_loss(lambdas, lambdas_loss_samples, dtimes_loss_samples)
            else:
                pred_type_logits, pred_time = self.hybrid_inference(latent_lambdas)
                self.loss = self.shuffle_hybrid_loss(pred_type_logits, pred_time)

            # marks
            marks_pred = self.mark_inference(latent_lambdas)
            marks_loss = self.mark_loss(marks_pred, self.marks_seq)

            self.loss = marks_loss

            # 4. train step
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.opt.minimize(self.loss)

            # assign prediction
            # shape -> [batch_size, max_len]
            self.pred_types = tf.argmax(pred_type_logits, axis=-1)
            # shape -> [batch_size, max_len]
            self.pred_time = pred_time
            self.pred_marks = marks_pred

    def mark_inference(self, latent_lambdas):
        n_gaussian = 3
        mark_dim = 1
        with tf.variable_scope('mark_inference'):
            mean_w = get_variable_weights('mark_mean_w', [self.hidden_dim, mark_dim, n_gaussian])
            var_w = get_variable_weights('mark_var_w', [self.hidden_dim, mark_dim, n_gaussian])
            weights = get_variable_bias('weights', [mark_dim, n_gaussian])

            # shape -> [..., mark_dim, n_gaussian]
            sample_shape = [1] * len(latent_lambdas.get_shape().as_list()) + [n_gaussian]
            samples = tf.random_normal(sample_shape)

            # shape -> [..., mark_dim, n_gaussian]
            means = tf.nn.relu(tensordot(latent_lambdas, mean_w))
            stds = tf.nn.relu(tensordot(latent_lambdas, var_w))
            mark_pred = means + stds * samples

            # shape -> [..., mark_dim, n_gaussian]
            scaled_weight = tf.reshape(tf.nn.softmax(weights, axis=-1), sample_shape)

            # shape -> [..., mark_dim]
            mark_pred = tf.reduce_sum(mark_pred * scaled_weight, axis=-1)

            self.means = means[0, -1, 0]
            self.stds = stds[0, -1, 0]
            self.weights = tf.nn.softmax(weights, axis=-1)

            self.mean = tf.reduce_mean(means)
            self.std = tf.reduce_mean(stds)

            return mark_pred

    # def mark_inference(self, latent_lambdas):
    #     mark_dim = 1
    #     with tf.variable_scope('mark_inference'):
    #         w = get_variable_weights('w', [self.hidden_dim, mark_dim])
    #         b = get_variable_weights('b', [mark_dim])
    #         return tensordot(latent_lambdas, w) + b

    def mark_loss(self, mark_pred, mark_label):
        seq_mask = self.seq_mask
        with tf.variable_scope('marks_loss'):
            # shape -> [batch_size, max_len - 1]
            seq_mask = seq_mask[:, 1:]

            mark_pred = mark_pred[:, :-1]
            mark_label = mark_label[:, 1:]

            mark_diff = tf.boolean_mask((mark_pred - mark_label) ** 2, seq_mask)
            mark_mse = tf.reduce_mean(mark_diff)

            mark_loss = mark_mse
            # mark_loss = mark_mse + 0.1 * (self.mean - self.std)
            return mark_loss

    # ---------------------- copy from AEHN --------------------------
    def embedding_layer(self, x_input):
        # add 1 dim because of EOS padding
        emb_layer = layers.Embedding(self.process_dim + 1, self.hidden_dim, name='type_embedding')
        # shape -> [batch_size, max_len, hidden_dim]
        emb = emb_layer(x_input)
        return emb

    def flow_layer(self, x_input, initial_state=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('flow_layer', reuse=reuse):
            rnn_layer = layers.LSTM(self.hidden_dim,
                                    return_state=True,
                                    return_sequences=True,
                                    name='rnn_layer')
        with tf.name_scope('flow_layer'):
            res = rnn_layer(x_input, initial_state=initial_state)
            output = res[0]

        return output

    def intensity_layer(self, x_input, reuse=tf.AUTO_REUSE):
        """ Compute the latent lambdas.
        :param x_input: [batch_size, max_len, hidden_dim]
        :param reuse:
        :return: [batch_size, max_len, hidden_dim]
        """
        with tf.variable_scope('intensity_layer', reuse=reuse):
            attention_layer = Attention(self.hidden_dim, 'general')
            delta_layer = layers.Dense(self.hidden_dim, activation=tf.nn.softplus, name='delta_layer')
            mu_layer = layers.Dense(self.hidden_dim, activation=tf.nn.tanh, name='mu_layer')

        with tf.name_scope('intensity_layer'):
            batch_size = tf.shape(x_input)[0]
            max_len = tf.shape(x_input)[1]
            # compute mu
            # shape -> [batch_size, max_len, hidden_dim]
            mus = mu_layer(x_input)

            # compute alpha
            # (batch_size, max_len, max_len, 1) (LowerTriangular)
            _, all_attention_weights = attention_layer.compute_attention_weight(x_input,
                                                                                x_input,
                                                                                x_input,
                                                                                pos_mask='right')
            # shape -> [batch_size, max_len, max_len, 1 | hidden_dim]
            # alphas = all_attention_weights
            alphas = all_attention_weights * x_input[:, None]

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

            # shape -> [batch_size, max_len, max_len, 1] (lower triangular: positive, upper: negative, diagonal: zero)
            base_elapses = tf.expand_dims(cum_dtimes[:, None, :] - cum_dtimes[:, :, None], axis=-1)

            # (dt_1, dt_2, ..., dt_0) (the last one is not used).
            # [batch_size, max_len]
            target_dtimes = tf.concat([self.dtimes_seq[:, 1:], self.dtimes_seq[:, :1]], axis=-1)

            # compute lambdas
            target_elapses = base_elapses + target_dtimes[:, :, None, None]
            # [batch_size, max_len, process_dim]
            lambdas = self.compute_latent_lambda(mus, alphas, deltas, target_elapses)

            if self.pred_method == 'loglikelihood':
                # use loop to avoid memory explode

                # general tensors
                # [max_len, batch_size, n_sample, hidden_dim]
                mus_trans = tf.transpose(mus, perm=[1, 0, 2])[:, :, None]
                # [max_len, batch_size, n_sample, max_len, hidden_dim]
                alphas_trans = tf.transpose(alphas, perm=[1, 0, 2, 3])[:, :, None]
                deltas_trans = tf.transpose(deltas, perm=[1, 0, 2, 3])[:, :, None]
                base_elapses_trans = tf.transpose(base_elapses, perm=[1, 0, 2, 3])[:, :, None]

                # sample lambdas for loss.
                dtimes_loss_samples = tf.linspace(start=0.0,
                                                  stop=1.0,
                                                  num=self.n_loss_integral_sample)
                # [batch_size, max_len, n_sample]
                dtimes_loss_samples = target_dtimes[:, :, None] * dtimes_loss_samples[None, None, :]
                # loop over max_len
                loss_scan_initializer = create_tensor([batch_size, self.n_loss_integral_sample, self.hidden_dim], 0.0)

                # [batch_size, max_len, n_loss_integral_sample, process_dim]
                lambdas_loss_samples = self.sample_lambda_by_scan(
                    self.get_compute_lambda_forward_fn(),
                    [
                        mus_trans,
                        alphas_trans,
                        deltas_trans,
                        base_elapses_trans,
                        swap_axes(dtimes_loss_samples[:, :, :, None, None], 0, 1)
                    ],
                    loss_scan_initializer
                )

                # sample lambdas for prediction.

                # [batch_size, max_len, n_sample]
                dtimes_pred_samples = tf.linspace(start=0.0,
                                                  stop=self.max_time_pred,
                                                  num=self.n_pred_integral_sample)[None, None, :]
                # use loop to avoid memory explode
                # loop over max_len
                pred_scan_initializer = create_tensor([batch_size, self.n_pred_integral_sample, self.hidden_dim], 0.0)

                # [batch_size, max_len, n_pred_integral_sample, process_dim]
                lambdas_pred_samples = self.sample_lambda_by_scan(
                    self.get_compute_lambda_forward_fn(),
                    [
                        mus_trans,
                        alphas_trans,
                        deltas_trans,
                        base_elapses_trans,
                        swap_axes(dtimes_pred_samples[:, :, :, None, None], 0, 1)
                    ],
                    pred_scan_initializer
                )
                return lambdas, lambdas_loss_samples, dtimes_loss_samples, lambdas_pred_samples, dtimes_pred_samples
            else:
                return lambdas, None, None, None, None

    def get_compute_lambda_forward_fn(self):
        compute_lambda_fn = self.compute_latent_lambda

        def forward_fn(acc, item):
            mu, alpha, delta, elapse, elapse_bias = item
            return compute_lambda_fn(mu, alpha, delta, elapse + elapse_bias)

        return forward_fn

    def compute_latent_lambda(self, mu, alpha, delta, elapse):
        """ compute imply lambda (mu + sum<alpha * exp(-delta * elapse)), and then transfer it to lambda with
        dimension process_dim.
        :param mu: [..., hidden_dim]
        :param alpha: [..., n_ob, 1(hidden_dim)]
        :param delta: [..., n_ob, 1(hidden_dim)]
        :param elapse: [..., n_ob, 1]
        :return:
        """

        # shape -> [..., hidden_dim]
        left_term = mu

        # to avoid nan calculated by exp after (nan * 0 = nan)
        elapse = tf.abs(elapse)
        # shape -> [..., hidden_dim]
        right_term = tf.reduce_sum(alpha * tf.exp(-delta * elapse), axis=-2)
        # shape -> [..., hidden_dim]
        imply_lambdas = left_term + right_term

        return imply_lambdas

    def compute_lambda(self, latent_lambda):
        with tf.variable_scope('lambda_layer', reuse=tf.AUTO_REUSE):
            lambda_w = tf.get_variable('lambda_w', shape=[self.hidden_dim, self.process_dim], dtype=tf.float32,
                                       initializer=tf.glorot_normal_initializer())
            lambda_b = tf.get_variable('lambda_b', shape=[self.process_dim], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
            return tf.nn.softplus(tensordot(latent_lambda, lambda_w) + lambda_b)

    def sample_lambda_by_scan(self, lambda_over_step_scan_fn, scan_elems, scan_initializer):
        lambdas_samples = tf.scan(
            lambda_over_step_scan_fn,
            scan_elems,
            initializer=scan_initializer)
        # [batch_size, max_len, ...]
        lambdas_samples = swap_axes(lambdas_samples, 1, 0)
        return lambdas_samples
