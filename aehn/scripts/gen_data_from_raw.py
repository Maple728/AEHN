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

from aehn.lib.utilities import concat_arrs_of_dict


def gen_data_from_synthetic_data(data_folder_name, data_csv_name):
    train_fname = '../data/{}/{}_training.csv'.format(data_folder_name, data_csv_name)
    valid_fname = '../data/{}/{}_validation.csv'.format(data_folder_name, data_csv_name)
    test_fname = '../data/{}/{}_testing.csv'.format(data_folder_name, data_csv_name)

    def process_file(fname, pkl_name):
        with open(fname, 'r') as file:
            lines = file.readlines()
            data = [np.array([float(item) for item in line.split(';')[0].split()]) for line in lines]

        res = dict()
        res['types'] = [np.array([0 for _ in record]) for record in data]
        res['timestamps'] = data

        pkl_file = '../data/{}/{}.pkl'.format(data_folder_name, pkl_name)
        with open(pkl_file, 'wb') as file:
            pickle.dump(res, file)

    process_file(train_fname, 'train')
    process_file(valid_fname, 'dev')
    process_file(test_fname, 'test')


def gen_data_from_retweet_or_so():
    folder = 'data_retweet'
    train_fname = folder + '/train.pkl'
    valid_fname = folder + '/valid.pkl'
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


def gen_data_from_order_book(n_points_per_record=32):
    fpath = 'amazon.csv'
    csv_data = np.loadtxt(fpath, delimiter=',', skiprows=1)

    def split_records(record, n_points):
        res = []
        n_records = len(record)

        for i in range(0, n_records, n_points):
            res.append(record[i: i + n_points])

        return res

    data = {
        'types': split_records(csv_data[:, 2], n_points_per_record),
        'timestamps': split_records(csv_data[:, 1], n_points_per_record),
        'marks': split_records(csv_data[:, 3], n_points_per_record)
    }

    n_records = len(data['types'])
    train_idx = int(0.6 * n_records)
    valid_idx = int(0.8 * n_records)

    print(train_idx, valid_idx, n_records)
    train_data = {
        'types': data['types'][:train_idx],
        'timestamps': data['timestamps'][:train_idx],
        'marks': data['marks'][:train_idx]
    }

    valid_data = {
        'types': data['types'][train_idx:valid_idx],
        'timestamps': data['timestamps'][train_idx:valid_idx],
        'marks': data['marks'][train_idx:valid_idx]
    }

    test_data = {
        'types': data['types'][valid_idx:],
        'timestamps': data['timestamps'][valid_idx:],
        'marks': data['marks'][valid_idx:]
    }

    with open('train' + '.pkl', 'wb') as file:
        pickle.dump(train_data, file)

    with open('valid' + '.pkl', 'wb') as file:
        pickle.dump(valid_data, file)

    with open('test' + '.pkl', 'wb') as file:
        pickle.dump(test_data, file)


if __name__ == '__main__':
    # gen_data_from_synthetic_data('data_poisson', 'poisson')
    # gen_data_from_synthetic_data('data_exphawkes', 'exponential_hawkes')
    # gen_data_from_synthetic_data('data_powerlaw_hawkes', 'powerlaw_hawkes')
    # gen_data_from_synthetic_data('data_selfcorrection', 'self_inhibiting')
    gen_data_from_order_book()
