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
def mse_tf(preds, labels):
    return tf.losses.mean_squared_error(labels, preds)


def mae_tf(preds, labels):
    return tf.losses.absolute_difference(labels, preds)


# ---------------- metric functions for numpy -----------------------
def type_acc_np(preds, labels):
    type_pred = preds['types']
    type_label = labels['types']

    return np.mean(type_pred == type_label)


def time_rmse_np(preds, labels):
    dt_pred = preds['dtimes']
    dt_label = labels['dtimes']

    dt_pred = np.reshape(dt_pred, [-1])
    dt_label = np.reshape(dt_label, [-1])

    rmse = np.sqrt(np.mean((dt_pred - dt_label) ** 2))
    return rmse


def time_mae_np(preds, labels):
    dt_pred = preds['dtimes']
    dt_label = labels['dtimes']

    dt_pred = np.reshape(dt_pred, [-1])
    dt_label = np.reshape(dt_label, [-1])

    mae = np.mean(np.abs(dt_pred - dt_label))
    return mae


def mape_np(preds, labels, threshold):
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    # zero mask
    mask = labels > threshold
    preds = preds[mask]
    labels = labels[mask]

    mape = np.mean(np.abs(preds - labels) / labels)
    return mape


def MdAE_np(preds, labels):
    """
    Median Absolute Error
    :param preds:
    :param labels:
    :return:
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    return np.median(np.abs(preds - labels))


def mase_np(preds, labels, benchmark_mae):
    """
    Mean Absolute Scaled Error
    :param preds:
    :param labels:
    :param benchmark_mae:
    :return:
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    mase = np.mean(np.abs(preds - labels) / benchmark_mae)
    return mase
