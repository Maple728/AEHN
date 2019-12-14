#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:55
@desc:
"""

import numpy as np
from abc import abstractmethod


class Scaler(object):
    """
    Scaler for scaling data into proper range.
    """

    @abstractmethod
    def fit(self, records):
        pass

    @abstractmethod
    def fit_scaling(self, records):
        pass

    @abstractmethod
    def scaling(self, records):
        pass

    @abstractmethod
    def inverse_scaling(self, scaled_records):
        pass


class MultipleScaler(Scaler):

    def __init__(self, **kwargs):
        self.scaler_dict = {}
        for k, scaler_class in kwargs.items():
            self.scaler_dict[k] = scaler_class()

    def fit(self, **kwargs):
        for k, records in kwargs.items():
            self.scaler_dict[k].fit(records)

    def fit_scaling(self, **kwargs):
        self.fit(**kwargs)
        return self.scaling(**kwargs)

    def scaling(self, **kwargs):
        ret = dict()
        for k, records in kwargs.items():
            ret[k] = self.scaler_dict[k].scaling(records)
        return ret

    def inverse_scaling(self, **kwargs):
        ret = dict()
        for k, scaled_records in kwargs.items():
            ret[k] = self.scaler_dict[k].inverse_scaling(scaled_records)
        return ret


class StandZeroMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._max_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._max_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return records / (self._max_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._max_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return scaled_records * (self._max_val + self._epsilon)


class MinMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = None
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._min_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=0)
        self._min_val = np.min(records, axis=0)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._min_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._min_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val


class ZeroMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = 0.0
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._min_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=0)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._min_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._min_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val