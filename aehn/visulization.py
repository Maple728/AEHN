#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2018/10/27 16:32
@desc:
"""
import argparse
import numpy as np
import yaml
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pltoff

from aehn.training.model_runner import ModelRunner
from aehn.preprocess.data_loader import DataLoader
from aehn.preprocess.data_provider import DataProvider

# GPU setting
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(args):
    config_filename = args.config_filename
    with open(config_filename) as config_file:
        config = yaml.load(config_file)
        data_config = config['data']

    # load data
    # get data source
    data_loader = DataLoader(data_config)
    train_ds, valid_ds, test_ds = data_loader.get_three_datasource()
    # get three data provider for model input
    data_config['batch_size'] = 1
    train_dp = DataProvider(train_ds, data_config)

    # get intensities
    with open(data_config['data_filename'].format('intensities'), 'rb') as file:
        intensities = pickle.load(file)

    with tf.Session() as sess:
        model_runner = ModelRunner(config)
        model_runner.restore_model(sess)
        model = model_runner.model
        i = 0
        for batch_data in train_dp.iterate_batch_data():
            # shape -> [batch_size, max_len, n_samples, process_dim]
            sampled_lambdas, sampled_dtimes = model.visualize_lambda(sess, batch_data)

            # visualize
            # sampled_lambdas = sampled_lambdas[:, :100]
            # sampled_dtimes = sampled_dtimes[:, :100]
            sampled_lambdas = np.reshape(np.sum(sampled_lambdas, axis=-1), [-1])

            offset = np.max(sampled_dtimes, axis=-1)
            offset[:, 1:] = offset[:, : -1]
            offset[:, 0] = 0
            offset = np.cumsum(offset, axis=-1)

            sampled_dtimes = np.reshape(sampled_dtimes + offset[:, :, None], [-1])

            # plt.figure(figsize=(5, 3))
            # modeled lambda
            mask_threshold = 35.0
            mask = sampled_dtimes < mask_threshold
            sampled_dtimes = sampled_dtimes[mask]
            sampled_lambdas = sampled_lambdas[mask]
            plt.plot(sampled_dtimes, sampled_lambdas, lw=1.5, c='b', label='AHEN')

            # real lambda
            intensity = intensities[0]
            x = np.array(range(len(intensity))) * 0.0907
            mask = x < mask_threshold

            x = x[mask]
            intensity = intensity[mask]
            plt.plot(x, intensity, lw=1.5, c='r', label='Hawkes')
            plt.xlabel('Time Index', {'size': 13})
            plt.ylabel('Intensity', {'size': 13})
            plt.legend()
            plt.show()

            # np.savetxt('aehn.csv', np.stack([sampled_dtimes, sampled_lambdas], axis=-1), delimiter=',')
            # np.savetxt('hawkes.csv', np.stack([x, intensity], axis=-1), delimiter=',')
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=sampled_dtimes,
            #                          y=sampled_lambdas,
            #                          mode='lines',
            #                          name='AEHN'))
            # fig.add_trace(go.Scatter(x=x,
            #                          y=intensity,
            #                          mode='lines',
            #                          name='real'))
            #
            # fig.update_layout(
            #     plot_bgcolor='white'
            # )
            # pltoff.plot(fig, filename='hawkes')
            # fig.show()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config_name = 'configs/bund.yaml'
    config_name = 'logs/[name-AEHN][proc-1][hidd-16][pred-loglikelihood]_0115162043/models/config-390.yaml'
    parser.add_argument('--config_filename', default=config_name, type=str, required=False,
                        help='Configuration filename for training or restoring the model.')
    args = parser.parse_args()
    main(args)
