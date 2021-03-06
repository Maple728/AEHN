#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np


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


def marks_rmse_np(preds, labels, **kwargs):
    seq_mask = kwargs['seq_mask']
    pred = preds['marks'][seq_mask]
    label = labels['marks'][seq_mask]

    # pred negative value mask
    pred[pred < 0] = 0

    rmse = np.sqrt(np.mean((pred - label) ** 2))
    return rmse


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
