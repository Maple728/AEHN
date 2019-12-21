#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: AEHN
@time: 2019/12/21 21:35
@desc:
"""

import pickle
import numpy as np

from lib.utils import concat_arrs_of_dict

def gen_data_from_self_inhibiting_data():
    train_fname = 'data_selfcorrection/self_inhibiting_training.csv'
    valid_fname = 'data_selfcorrection/self_inhibiting_validation.csv'
    test_fname = 'data_selfcorrection/self_inhibiting_testing.csv'

    def process_file(fname):
        with open(fname, 'r') as file:
            lines = file.readlines()

            ''.split()
            data = [np.array([float(item) for item in line.split()]) for line in lines]

        res = dict()
        res['types'] = [np.array([0 for _ in record]) for record in data]
        res['timestamps'] = data

        with open(fname + '.pkl', 'wb') as file:
            pickle.dump(res, file)

    process_file(train_fname)
    process_file(valid_fname)
    process_file(test_fname)


def gen_data_from_retweet_or_so():
    folder = 'data_retweet'
    train_fname = folder + '/train.pkl'
    valid_fname = folder + '/dev.pkl'
    test_fname = folder + '/test.pkl'

    with open(train_fname, 'rb') as f:
        train_records = pickle.load(f)['train']

    with open(valid_fname, 'rb') as f:
        valid_records = pickle.load(f)['dev']

    with open(test_fname, 'rb') as f:
        test_records = pickle.load(f)['test']

    data = concat_arrs_of_dict([train_records, valid_records, test_records])

    n_records = len(data['types'])
    train_idx = int(0.6 * n_records)
    valid_idx = int(0.8 * n_records)

    print(train_idx, valid_idx, n_records)
    train_data = {
        'types': data['types'][:train_idx],
        'timestamps': data['timestamps'][:train_idx]
    }

    valid_data = {
        'types': data['types'][train_idx:valid_idx],
        'timestamps': data['timestamps'][train_idx:valid_idx]
    }

    test_data = {
        'types': data['types'][valid_idx:],
        'timestamps': data['timestamps'][valid_idx:]
    }

    with open('train' + '.pkl', 'wb') as file:
        pickle.dump(train_data, file)

    with open('valid' + '.pkl', 'wb') as file:
        pickle.dump(valid_data, file)

    with open('test' + '.pkl', 'wb') as file:
        pickle.dump(test_data, file)


if __name__ == '__main__':
    gen_data_from_retweet_or_so()
