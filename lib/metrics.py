#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np
import tensorflow as tf


# ----------------- loss function for tensorflow --------------------
# loss function suffix with '_tf'
def shuffle_label_hybrid_loss_tf(lambdas, pred_times,
                                 label_types, label_times,
                                 seq_mask,
                                 **kwargs):
    """ Calculate the total loss of all points in the sequence by using the latter point as the label predicted by the
    previous point.
    :param lambdas: [batch_size, max_len, process_dim]
    :param pred_times: [batch_size, max_len]
    :param label_types: [batch_size, max_len]
    :param label_times: [batch_size, max_len]
    :param seq_mask: [batch_size, max_len]
    :return:
    """
    with tf.variable_scope('loss'):
        # shape -> [batch_size, max_len - 1]
        seq_mask = seq_mask[:, 1:]
        # (batch_size, max_len - 1, process_dim)
        type_label = label_types[:, 1:]

        pred_type_logits = lambdas[:, :-1]
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


def shuffle_label_loglikelihood_loss_tf(lambdas, lambdas_samples, seq_event_onehot, dtimes_seq):
    """

    :param lambdas: [batch_size, max_len, process_dim]
    :param lambdas_samples: [batch_size, max_len, n_sample, process_dim]
    :param seq_event_onehot: [batch_size, max_len, process_dim]
    :param dtimes_seq: [batch_size, max_len]
    :return:
    """
    # shape -> [batch_size, max_len - 1]
    target_lambdas = tf.reduce_sum(lambdas[:, :-1] * seq_event_onehot[:, 1:], axis=-1)
    target_lambdas_masked = tf.boolean_mask(target_lambdas, target_lambdas > 0)
    term_1 = tf.reduce_sum(tf.log(target_lambdas_masked))

    # shape -> [batch_size, max_len - 1, 1]
    lambdas_total_mask = tf.reduce_sum(seq_event_onehot[:, 1:], axis=-1, keepdims=True)
    # shape -> [batch_size, max_len - 1, n_sample]
    lambdas_total_samples = tf.reduce_sum(lambdas_samples[:, :-1], axis=-1)
    # shape -> [batch_size, max_len - 1]
    lambdas_integral = tf.reduce_mean(lambdas_total_samples, axis=-1) * dtimes_seq[:, 1:]

    term_2 = tf.reduce_sum(lambdas_total_mask * lambdas_integral)

    return - (term_1 - term_2)


# ---------------- metric functions for numpy -----------------------
# metric function suffix with '_np'
def type_acc_np(preds, labels, **kwargs):
    seq_mask = kwargs['seq_mask']

    type_pred = preds['types'][seq_mask]
    type_label = labels['types'][seq_mask]


    return np.mean(type_pred == type_label)


def time_rmse_np(preds, labels, **kwargs):
    seq_mask = kwargs['seq_mask']
    dt_pred = preds['dtimes'][seq_mask]
    dt_label = labels['dtimes'][seq_mask]

    rmse = np.sqrt(np.mean((dt_pred - dt_label) ** 2))
    return rmse


def time_mae_np(preds, labels, **kwargs):
    seq_mask = kwargs['seq_mask']
    dt_pred = preds['dtimes'][seq_mask]
    dt_label = labels['dtimes'][seq_mask]

    dt_pred = np.reshape(dt_pred, [-1])
    dt_label = np.reshape(dt_label, [-1])

    mae = np.mean(np.abs(dt_pred - dt_label))
    return mae


def mape_np(preds, labels, **kwargs):
    """

    :param preds:
    :param labels:
    :param kwargs: need (threshold)
    :return:
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    threshold = kwargs['threshold']

    # zero mask
    mask = labels > threshold
    preds = preds[mask]
    labels = labels[mask]

    mape = np.mean(np.abs(preds - labels) / labels)
    return mape
