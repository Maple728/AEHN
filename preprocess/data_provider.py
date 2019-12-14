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
        return [target_data['types'], target_data['dtimes']]

    def iterate_batch_data(self, provide_label=True):
        """ Get batch model input of one epoch.
        :param provide_label: return values with label if True
        :return:
        """
        # record_data of a partition whose shape is [n_records, ...]
        for feat_data, target_data in self._data_source.load_partition_data():
            # yield feat_data and target_data to batch data separately
            datas = self._process_model_input(feat_data, target_data, False)

            yield yield2batch_data(datas, self._batch_size, keep_remainder=True)


