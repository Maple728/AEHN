#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: AEHN
@time: 2019/12/21 21:35
@desc:
"""
from functools import reduce
import pickle
import pandas as pd
import numpy as np
import os

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


def gen_data_from_nhp_format(data_folder):
    train_fname = data_folder + '/train.pkl'
    valid_fname = data_folder + '/dev.pkl'
    test_fname = data_folder + '/test.pkl'

    with open(train_fname, 'rb') as f:
        train_records = pickle.load(f)['train']

    with open(valid_fname, 'rb') as f:
        valid_records = pickle.load(f)['dev']

    with open(test_fname, 'rb') as f:
        test_records = pickle.load(f)['test']

    with open('train' + '.pkl', 'wb') as file:
        pickle.dump(train_records, file)

    with open('valid' + '.pkl', 'wb') as file:
        pickle.dump(valid_records, file)

    with open('test' + '.pkl', 'wb') as file:
        pickle.dump(test_records, file)


def split_records(record, n_points):
    res = []
    n_records = len(record)

    for i in range(1, n_records, n_points):
        # i - 1 to avoid the last point doesn't be evaluated.
        res.append(record[i - 1: i + n_points])

    return res


def gen_data_from_order_book(tick, n_points_per_record=32, folder='raw_data'):
    fpath = os.path.join(folder, f'{tick}.csv')
    csv_data = np.loadtxt(fpath, delimiter=',', skiprows=1)


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
    return train_data, valid_data, test_data


def gen_data_from_multi_order_book(ticks, n_points_per_record=32, folder='raw_data'):
    train_data = []
    valid_data = []
    test_data = []

    for tick in ticks:
        t, v, e = gen_data_from_order_book(tick, n_points_per_record, folder)
        train_data.append(t)
        valid_data.append(v)
        test_data.append(e)

    def concat_dict_list(dict_list):
        res = {}
        for key in dict_list[0].keys():
            res[key] = np.concatenate([d[key] for d in dict_list], axis=0)

        return res

    train_data = concat_dict_list(train_data)
    valid_data = concat_dict_list(valid_data)
    test_data = concat_dict_list(test_data)

    with open('train' + '.pkl', 'wb') as file:
        pickle.dump(train_data, file)

    with open('valid' + '.pkl', 'wb') as file:
        pickle.dump(valid_data, file)

    with open('test' + '.pkl', 'wb') as file:
        pickle.dump(test_data, file)


def make_fin_data(tick, folder='raw_data'):
    file = os.path.join(folder, f'{tick}_2012-06-21_34200000_57600000_message_1.csv')

    data_df = pd.read_csv(file, header=None)

    data_df.columns = ['timestamps', 'types', 'orderid', 'size', 'price', 'direction']

    # merge same type of event happening at the same time
    data_df = data_df.groupby(['timestamps', 'types'])['size'].agg('sum').reset_index()

    # remove first type 3 and then type 1 at the same time
    data_df.drop_duplicates('timestamps', inplace=True)

    data_df.to_csv(os.path.join(folder, f'{tick}.csv'), header=True)

    return


def truncate_data2fix_window(data_folder, window_size):

    def split_data(data):
        res = {}
        for k, v in data.items():
            res[k] = reduce(lambda acc, x: acc + x, [split_records(record, window_size) for record in v], [])
        return res

    train_fname = data_folder + '/train.pkl'
    valid_fname = data_folder + '/valid.pkl'
    test_fname = data_folder + '/test.pkl'

    with open(train_fname, 'rb') as f:
        train_records = pickle.load(f)

    with open(valid_fname, 'rb') as f:
        valid_records = pickle.load(f)

    with open(test_fname, 'rb') as f:
        test_records = pickle.load(f)

    with open('train' + '.pkl', 'wb') as file:
        data = split_data(train_records)
        print(len(data['types']))
        pickle.dump(data, file)

    with open('valid' + '.pkl', 'wb') as file:
        data = split_data(valid_records)
        print(len(data['types']))
        pickle.dump(data, file)

    with open('test' + '.pkl', 'wb') as file:
        data = split_data(test_records)
        print(len(data['types']))
        pickle.dump(data, file)


if __name__ == '__main__':
    # gen_data_from_synthetic_data('data_poisson', 'poisson')
    # gen_data_from_synthetic_data('data_exphawkes', 'exponential_hawkes')
    # gen_data_from_synthetic_data('data_powerlaw_hawkes', 'powerlaw_hawkes')
    # gen_data_from_synthetic_data('data_selfcorrection', 'self_inhibiting')
    # gen_data_from_order_book()

    # ticks = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']
    # [make_fin_data(tick) for tick in ticks]
    # gen_data_from_multi_order_book(ticks)
    truncate_data2fix_window('../data/data_retweet_trunc', window_size=64)
