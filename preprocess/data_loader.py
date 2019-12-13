#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:42
@desc:
"""

from preprocess.scaler import Scaler
from preprocess.data_source import DataSource
from lib.utils import get_metric_functions


def get_metrics_callback_from_names(metric_names):
    metric_functions = get_metric_functions(metric_names)

    def metrics(preds, labels):
        res = dict()
        for metric_name, metric_func in zip(metric_names, metric_functions):
            res[metric_name] = metric_func(preds, labels)
        return res

    return metrics


def get_static_data_callback(data):
    """
    callback generator for getting data online
    :param data:
    :return:
    """

    def data_callback():
        yield data

    return data_callback


class DataLoader(object):

    def __init__(self, data_name, data_filename, metrics, cache_dir,
                 **kwargs):
        self._data_name = data_name
        self._data_filename = data_filename
        self._cache_dir = cache_dir
        self._metrics = get_metrics_callback_from_names(metrics)

    def get_three_datasource(self):
        """ Load the raw data, and then return three data sources containing train data, validation and test
        data separately.
        :return: train, validation and test DataSource.
        """
        # load data
        # shape -> [n_records, ...]
        records = None

        # split datasets
        train_records, valid_records, test_records = records, records, records

        # scaling feat series
        feat_scaler = Scaler()
        train_feats = feat_scaler.fit_scaling(train_records)
        valid_feats = feat_scaler.scaling(valid_records)
        test_feats = feat_scaler.scaling(test_records)

        # scaling target series
        tgt_scaler = Scaler()
        train_tgts = tgt_scaler.fit_scaling(train_records)
        valid_tgts = tgt_scaler.scaling(valid_records)
        test_tgts = tgt_scaler.scaling(test_records)

        # wrapping data into DataSource
        train_ds = DataSource(self._data_name + '_train',
                              metric_callback=self._metrics,
                              retrieve_data_callback=get_static_data_callback([train_feats, train_tgts]),
                              scaler=tgt_scaler, cache_dir=self._cache_dir)
        valid_ds = DataSource(self._data_name + '_valid',
                              metric_callback=self._metrics,
                              retrieve_data_callback=get_static_data_callback([valid_feats, valid_tgts]),
                              scaler=tgt_scaler, cache_dir=self._cache_dir)
        test_ds = DataSource(self._data_name + '_test',
                             metric_callback=self._metrics,
                             retrieve_data_callback=get_static_data_callback([test_feats, test_tgts]),
                             scaler=tgt_scaler, cache_dir=self._cache_dir)
        return train_ds, valid_ds, test_ds
