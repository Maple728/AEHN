#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/5 21:01
@desc:
"""
from abc import abstractmethod
import numpy as np

from aehn.lib import yield2batch_data, get_metrics_callback_from_names
from aehn.lib import DictScaler, VoidScaler, ZeroMaxScaler, StandZeroMaxScaler


class AbstractDataProvider(object):

    @abstractmethod
    def iterate_batch_data(self):
        """ Get batch model input of one epoch.
        Remark: batch -> partition -> epoch
        :return: yield a list containing batch inputs until the end of the epoch.
        """
        pass

    @abstractmethod
    def get_metrics(self, preds, labels):
        """ Calculate the metrics of preds and labels.
        :param preds:
        :param labels:
        :return: a dictionary of the metrics.
        """
        pass

    @abstractmethod
    def epoch_inverse_scaling(self, scaled_records):
        """ Inverse the scaled_records to real scale.
        :param scaled_records:
        :return: real scale records.
        """
        pass


class DataProvider(AbstractDataProvider):
    """
    Data provider for processing model inputs.
    """
    def __init__(self, data_source, data_config, scaler=None):
        self._data_source = data_source
        self._batch_size = data_config['batch_size']
        self._metrics_function = get_metrics_callback_from_names(data_config['metrics'])
        self._scaler = scaler if scaler else DictScaler(dtimes=VoidScaler)
        self._is_first_iterate = True

        self._type_padding = data_config['process_dim']

    def get_scaler(self):
        return self._scaler

    def epoch_inverse_scaling(self, scaled_records):
        return self._scaler.inverse_scaling(scaled_records)

    def get_metrics(self, preds, labels):
        seq_mask = labels['types'] < self._type_padding
        return self._metrics_function(preds, labels, seq_mask=seq_mask)

    def iterate_batch_data(self):
        # record_data of a partition whose shape is [n_records, ...]
        for data in self._data_source.load_partition_data():
            if self._is_first_iterate:
                data_stats = self._dataset_statistics(data)
                print(f'Load dataset {self._data_source.data_name}: {data_stats}')

            inputs = self._process_model_input(data)
            if self._scaler.is_fit():
                scaled_inputs = self._scaler.scaling(inputs)
            else:
                scaled_inputs = self._scaler.fit_scaling(inputs)

            # yield records to batch data separately
            for batch_data in yield2batch_data(scaled_inputs, self._batch_size, keep_remainder=True):
                yield batch_data

        if self._is_first_iterate:
            self._is_first_iterate = False

    def _process_model_input(self, records):
        """ Process each item as model input.
        :param records:
        :return:
        """
        type_padding = self._type_padding
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
            dt_seqs_padded[i, : len_seq] = dt_seqs[i]

        ret = dict()
        ret['types'] = type_seqs_padded
        ret['dtimes'] = dt_seqs_padded

        return ret

    def _dataset_statistics(self, data):
        statistics = {}
        # get target seqs
        type_seqs = data['types']
        dt_seqs = [[t_seq[i] - t_seq[max(i - 1, 0)] for i in range(len(t_seq))]
                   for t_seq in data['timestamps']]

        event_num = self._type_padding

        types = sum(type_seqs, [])
        dts = sum(dt_seqs, [])

        # get statistics
        statistics['n_records'] = len(type_seqs)
        statistics['max_len_of_record'] = max([len(seq) for seq in type_seqs])
        statistics['min_len_of_record'] = min([len(seq) for seq in type_seqs])
        statistics['max_dt'] = np.max(dts)
        statistics['min_dt'] = np.min(dts)
        statistics['mean_dt'] = np.mean(dts)
        statistics['median_dt'] = np.median(dts)

        type_count = [0] * event_num
        for t in types:
            type_count[t] += 1

        type_ratio = np.divide(type_count, np.sum(type_count))
        statistics['max_type_ratio'] = np.max(type_ratio)

        return statistics
