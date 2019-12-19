#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 15:43
@desc:
"""
import os
from functools import reduce
from operator import mul

from lib.metrics import *


def get_tf_loss_function(loss_name):
    """ Get tensorflow loss function by loss_name
    :param loss_name:
    :return:
    """
    return eval(loss_name + '_tf')


def get_metric_functions(metric_name_list):
    """ Get metric functions from a list of metric name.
    :param metric_name_list:
    :return:
    """
    metric_functions = []
    for metric_name in metric_name_list:
        metric_functions.append(eval(metric_name + '_np'))
    return metric_functions


def get_metrics_callback_from_names(metric_names):
    metric_functions = get_metric_functions(metric_names)

    def metrics(preds, labels, **kwargs):
        """
        :param preds:
        :param labels:
        :return:
        """
        res = dict()
        for metric_name, metric_func in zip(metric_names, metric_functions):
            res[metric_name] = metric_func(preds, labels, **kwargs)
        return res

    return metrics


def get_num_trainable_params():
    """ Get the number of trainable parameters in current session (model).
    :return:
    """
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def set_random_seed(seed=9899):
    """ set random seed for numpy and tensorflow
    :param seed:
    :return:
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_config_string(config):
    """ Generate a name for config.
    :param config:
    :return:
    """
    key_len = 4
    str_config = ''
    for k, v in config.items():
        str_config += k[:key_len] + '-' + str(v) + '_'
    return str_config[:-1]


def window_rolling(origin_data, window_size):
    """Rolling data over 0-dim.
    :param origin_data: ndarray of [n_records, ...]
    :param window_size: window_size
    :return: [n_records - window_size + 1, window_size, ...]
    """
    n_records = len(origin_data)
    if n_records < window_size:
        return None

    data = origin_data[:, None]
    all_data = []
    for i in range(window_size):
        all_data.append(data[i: (n_records - window_size + i + 1)])

    # shape -> [n_records - window_size + 1, window_size, ...]
    rolling_data = np.hstack(all_data)

    return rolling_data


def yield2batch_data(arr_dict, batch_size, keep_remainder=True):
    """Iterate the dictionary of array over 0-dim to get batch data.
    :param arr_dict: a dictionary containing array whose shape is [n_items, ...]
    :param batch_size:
    :param keep_remainder: Discard the remainder if False, otherwise keep it.
    :return:
    """
    if arr_dict is None or len(arr_dict) == 0:
        return

    keys = list(arr_dict.keys())

    idx = 0
    n_items = len(arr_dict[keys[0]])
    while idx < n_items:
        if idx + batch_size > n_items and keep_remainder is False:
            return
        next_idx = min(idx + batch_size, n_items)

        yield {k: arr_dict[k][idx: next_idx] for k in keys}

        # update idx
        idx = next_idx


def create_folder(*args):
    """Create path if the folder doesn't exist.
    :param args:
    :return: The folder's path depends on operating system.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def concat_arrs_of_dict(dict_list):
    res = dict()

    keys = dict_list[0].keys()
    for k in keys:
        arr_list = []
        for d in dict_list:
            arr_list.append(d[k])
        res[k] = np.concatenate(arr_list, axis=0)

    return res
