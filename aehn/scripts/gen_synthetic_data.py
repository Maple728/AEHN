import numpy as np
import pickle


def fuse_node_times(e):
    n_nodes = len(e)
    type_list = [i * np.ones_like(e[i], dtype=int) for i in range(n_nodes)]

    concat_times = np.concatenate(e)
    sorting_index = concat_times.argsort()
    type_arr = np.concatenate(type_list)[sorting_index]
    return concat_times[sorting_index], type_arr


def multi_hawkes_simulate(adjacency, decays, baseline, end_time, n_simulations):
    from tick import hawkes
    timestamps_seqs = []
    intensities_seqs = []
    for i in range(n_simulations):
        hawkes_kernel = hawkes.SimuHawkesExpKernels(adjacency=adjacency,
                                                    decays=decays,
                                                    baseline=baseline,
                                                    end_time=end_time)
        hawkes_kernel.track_intensity(1.0)
        hawkes_kernel.simulate()

        intensities = np.sum(np.stack(hawkes_kernel.tracked_intensity, axis=-1), axis=-1, keepdims=False)

        timestamps_seqs.append(hawkes_kernel.timestamps)
        intensities_seqs.append(intensities)

    timestamps = [fuse_node_times(e) for e in timestamps_seqs]
    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res, intensities_seqs


def make_2d_hawkes():
    from tick import hawkes
    adj_2d = np.array([[0.1, 0.1],
                       [0.01, 0.2]])
    decay_2d = 1
    baseline_2d = np.array([0.3, 0.1])

    tmax = 250

    hawkes_2d = hawkes.SimuHawkesExpKernels(adj_2d,
                                            decay_2d,
                                            baseline=baseline_2d,
                                            end_time=tmax,
                                            verbose=False)

    hawkes_2d.reset()
    hawkes_2d.track_intensity(0.1)
    # hawkes_2d.simulate()

    multi_2d = hawkes.SimuHawkesMulti(hawkes_2d,
                                      n_simulations=500,
                                      n_threads=4)

    multi_2d.simulate()

    timestamps = [fuse_node_times(e) for e in multi_2d.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    print('length of timestamps {}'.format(len(event_timestamps[0])))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_1d_hawkes():
    tmax = 140
    res, intensities = multi_hawkes_simulate(adjacency=np.array([[0.5]]),
                                             decays=1.6,
                                             baseline=np.array([0.5]),
                                             end_time=tmax,
                                             n_simulations=500)

    return res, intensities


def make_3d_hawkes():
    from tick import hawkes
    adj_3d = np.array([[0.1, 0.1, 0.5],
                       [0.01, 0.2, 0.02],
                       [0.03, 0.0, 0.5]])
    decay_3d = 0.5
    baseline_3d = np.array([0.1, 0.15, 0.4])

    tmax = 100

    hawkes_3d = hawkes.SimuHawkesExpKernels(adj_3d,
                                            decay_3d,
                                            baseline=baseline_3d,
                                            end_time=tmax,
                                            verbose=False)

    # hawkes_3d.reset()
    hawkes_3d.track_intensity(0.1)
    hawkes_3d.simulate()

    multi_3d = hawkes.SimuHawkesMulti(hawkes_3d,
                                      n_simulations=500,
                                      n_threads=4)

    multi_3d.simulate()

    timestamps = [fuse_node_times(e) for e in multi_3d.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_10d_hawkes():
    from tick import hawkes
    adj = np.array([[0.05] * 10,
                    [0.13 - i / 40 for i in range(10)],
                    [0.05 - i / 40 for i in range(10)],
                    [0.03 * i / 20 for i in range(10)],
                    [0.1 - i / 30 for i in range(10)],
                    [0.09] * 10,
                    [0.5 * (i + 1) / 20 for i in np.arange(start=0, stop=20, step=2)],
                    [0.0] * 10,
                    [0.09] * 5 + [0.01] * 5,
                    [0.7 - i / 40 for i in range(10)]]) * 0.5

    decay = 0.05
    baseline = np.array([0.7, 0.15, 0.4, 0.2, 0.5] * 2)

    tmax = 30

    hawkes_kernel = hawkes.SimuHawkesExpKernels(adj,
                                                decay,
                                                baseline=baseline,
                                                end_time=tmax,
                                                verbose=False)

    hawkes_kernel.reset()
    hawkes_kernel.track_intensity(0.1)

    multi_hawkes = hawkes.SimuHawkesMulti(hawkes_kernel,
                                          n_simulations=500,
                                          n_threads=4)

    multi_hawkes.simulate()

    timestamps = [fuse_node_times(e) for e in multi_hawkes.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_10d_hawkes_2():
    from tick import hawkes
    adj = np.array([[0.05] * 10,
                    [0.03] * 5 + [0.07] * 5,
                    [0.05 + i / 40 for i in range(10)],
                    [0.03 * i / 20 for i in range(10)],
                    [0.013] * 2 + [0.021] * 8,
                    [0.09] * 10,
                    [0.5 * (i + 1) / 20 for i in np.arange(start=0, stop=10, step=1)],
                    [0.0] * 10,
                    [0.09] * 4 + [0.01] * 6,
                    [0.08 - i / 600 for i in range(10)]])

    decay = 0.2
    baseline = np.array([0.7, 0.25, 0.4, 0.2, 0.5] * 2)

    tmax = 30

    hawkes_kernel = hawkes.SimuHawkesExpKernels(adj,
                                                decay,
                                                baseline=baseline,
                                                end_time=tmax,
                                                verbose=False)

    hawkes_kernel.reset()
    hawkes_kernel.track_intensity(0.1)

    multi_hawkes = hawkes.SimuHawkesMulti(hawkes_kernel,
                                          n_simulations=500,
                                          n_threads=4)

    multi_hawkes.simulate()

    timestamps = [fuse_node_times(e) for e in multi_hawkes.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_5d_hawkes():
    from tick import hawkes
    adj = np.array([[0.1, 0.1, 0.5, 0.6, 0.1],
                    [0.1, 0.2, 0.02, 0.04, 0.2],
                    [0.3, 0.0, 0.5, 0.4, 0.1],
                    [0.01, 0.4, 0.05, 0.02, 0.1],
                    [0.03, 0.3, 0.5, 0.4, 0.1]])
    decay = 0.4
    baseline = np.array([0.1, 0.15, 0.4, 0.1, 0.5])

    tmax = 20

    hawkes_kernel = hawkes.SimuHawkesExpKernels(adj,
                                                decay,
                                                baseline=baseline,
                                                end_time=tmax,
                                                verbose=False)

    hawkes_kernel.reset()
    hawkes_kernel.track_intensity(0.1)

    multi_hawkes = hawkes.SimuHawkesMulti(hawkes_kernel,
                                          n_simulations=500,
                                          n_threads=4)

    multi_hawkes.simulate()

    timestamps = [fuse_node_times(e) for e in multi_hawkes.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_20d_hawkes():
    from tick import hawkes
    adj = np.array([[0.05] * 20,
                    [0.03] * 5 + [0.07] * 15,
                    [0.05 + i / 40 for i in range(20)],
                    [0.03 * i / 20 for i in range(20)],
                    [0.013] * 9 + [0.021] * 11,
                    [0.09] * 20,
                    [0.5 * (i + 1) / 20 for i in np.arange(start=0, stop=20, step=1)],
                    [0.0] * 20,
                    [0.09] * 10 + [0.01] * 10,
                    [0.08 - i / 600 for i in range(20)],
                    [0.05] * 20,
                    [(40 - i) / 200 for i in range(20)],
                    [0.05 + i / 40 for i in range(20)],
                    [0.03 * i / 20 for i in range(20)],
                    [0.1 + i / 30 for i in range(20)],
                    [0.09] * 18 + [0.04] * 2,
                    [0.01] * 20,
                    [0.0] * 20,
                    [0.09] * 10 + [0.01] * 10,
                    [0.07 - i / 400 for i in range(20)]
                    ]) * 0.5

    decay = 0.1
    baseline = np.array([0.7, 0.25, 0.4, 0.2, 0.5] * 4)

    tmax = 15

    hawkes_kernel = hawkes.SimuHawkesExpKernels(adj,
                                                decay,
                                                baseline=baseline,
                                                end_time=tmax,
                                                verbose=False)

    hawkes_kernel.reset()
    hawkes_kernel.track_intensity(0.1)

    multi_hawkes = hawkes.SimuHawkesMulti(hawkes_kernel,
                                          n_simulations=500,
                                          n_threads=4)

    multi_hawkes.simulate()

    timestamps = [fuse_node_times(e) for e in multi_hawkes.timestamps]

    event_timestamps, event_types = list(zip(*timestamps))

    res = dict()
    res['timestamps'] = event_timestamps
    res['types'] = event_types

    return res


def make_syn_dataset(dim=2):
    if dim == 1:
        data = make_1d_hawkes()
    elif dim == 2:
        data = make_2d_hawkes()
    elif dim == 3:
        data = make_3d_hawkes()
    elif dim == 5:
        data = make_5d_hawkes()
    elif dim == 10:
        data = make_10d_hawkes_2()
    else:
        data = make_20d_hawkes()

    ds, intensities = data
    data = ds

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

    with open('intensities' + '.pkl', 'wb') as file:
        pickle.dump(intensities, file)


if __name__ == '__main__':
    make_syn_dataset(dim=1)
    # make_syn_dataset(dim=10)
