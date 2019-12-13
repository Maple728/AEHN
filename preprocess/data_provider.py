#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/5 21:01
@desc:
"""

from lib.utils import window_rolling, yield2batch_data


class DataProvider:
    """
    Data provider for processing model inputs.
    """
    def __init__(self, data_source, T, n, T_skip, horizon, batch_size, **kwargs):
        self._data_source = data_source
        self._batch_size = batch_size
        self._T = T
        self._n = n
        self._T_skip = T_skip
        self._horizon = horizon

    @property
    def data_source(self):
        return self._data_source

    def _process_model_input(self, feat_data, target_data, provide_label):
        """ Process each item as model input.

        :param feat_data: [n_items, window_size, ...]
        :param target_data: [n_items, window_size, ...]
        :param provide_label:
        :return: feat (and label if provide_label): shape -> [n_items, ...]
        """
        x = None
        label = None
        if provide_label:
            return x, label
        else:
            return x

    def iterate_batch_data(self, provide_label=True):
        """ Get batch model input of one epoch.
        :param provide_label: return values with label if True
        :return:
        """
        if provide_label:
            window_size = self._T_skip * self._n + self._T + self._horizon
        else:
            window_size = self._T_skip * self._n + self._T

        # record_data of a partition whose shape is [n_records, ...]
        for feat_data, target_data in self._data_source.load_partition_data():
            # yield feat_data and target_data to batch data separately

            # process
            n_items = len(target_data)
            offset = window_size + self._batch_size - 1
            idx = offset
            while idx < n_items:
                # shape -> [batch_size, window_size, D]
                batch_target_data = window_rolling(target_data[idx - offset:idx], window_size)

                yield self._process_model_input(batch_target_data, batch_target_data, provide_label)
                idx = min(idx + self._batch_size, n_items) if idx < n_items else n_items + 1

