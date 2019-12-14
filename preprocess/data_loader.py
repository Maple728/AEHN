#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:42
@desc:
"""
import pickle
import numpy as np

from preprocess.scaler import MultipleScaler, ZeroMaxScaler
from preprocess.data_source import DataSource
from lib.utils import get_metric_functions


def get_metrics_callback_from_names(metric_names, type_padding):
    metric_functions = get_metric_functions(metric_names)

    def metrics(preds, labels):
        """
        :param preds: []
        :param labels:
        :return:
        """
        seqs_mask = labels['types'] < type_padding
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
                 process_dim,
                 **kwargs):
        self._data_name = data_name
        self._data_filename = data_filename
        self._cache_dir = cache_dir
        self._metrics = get_metrics_callback_from_names(metrics)
        self._process_dim = process_dim

    def get_three_datasource(self):
        """ Load the raw data, and then return three data sources containing train data, validation and test
        data separately.
        :return: train, validation and test DataSource.
        """
        # load data
        # keys(types, timesteps)   format:n_seqs * [seq_len]
        with open(self._data_filename.format('train'), 'rb') as f:
            train_records = pickle.load(f)

        with open(self._data_filename.format('dev'), 'rb') as f:
            valid_records = pickle.load(f)

        with open(self._data_filename.format('test'), 'rb') as f:
            test_records = pickle.load(f)

        # process data
        train_records = self.process_records_with_padding(train_records)
        valid_records = self.process_records_with_padding(valid_records)
        test_records = self.process_records_with_padding(test_records)

        # scaling feat series
        train_feats = None
        valid_feats = None
        test_feats = None

        # scaling target series
        tgt_scaler = MultipleScaler(time=ZeroMaxScaler)
        train_tgts = tgt_scaler.fit_scaling(time=train_records['dtime'])
        valid_tgts = tgt_scaler.scaling(time=valid_records['dtime'])
        test_tgts = tgt_scaler.scaling(time=test_records['dtime'])

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

    def process_records_with_padding(self, records):
        type_padding = self._process_dim
        dt_padding = 0.0

        type_seqs = records['types']
        time_seqs = records['timestamps']
        # dt_i = t_i - t_{i-1}
        dt_seqs = [[t_seq[i] - t_seq[max(i - 1, 0)] for i in range(len(t_seq))]
                   for t_seq in time_seqs]

        n_records = len(type_seqs)
        max_len = max([len(seq) for seq in type_seqs])

        # padding
        type_seqs_padded = np.ones([n_records, max_len]) * type_padding
        dt_seqs_padded = np.ones([n_records, max_len]) * dt_padding

        for i in range(n_records):
            len_seq = len(type_seqs[i])
            type_seqs_padded[i, :len_seq] = type_seqs[i]
            dt_seqs_padded[i, : len_seq] = dt_seqs

        ret = dict()
        ret['types'] = type_seqs_padded
        ret['dtimes'] = dt_seqs_padded
        return ret
