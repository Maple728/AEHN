import numpy as np
import pickle


def fuse_node_times(e):
    n_nodes = len(e)
    type_list = [i * np.ones_like(e[i], dtype=int) for i in range(n_nodes)]

    concat_times = np.concatenate(e)
    sorting_index = concat_times.argsort()
    type_arr = np.concatenate(type_list)[sorting_index]
    return concat_times[sorting_index], type_arr


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
    from tick import hawkes
    tmax = 180
    hawkes_1d = hawkes.SimuHawkes(n_nodes=1, end_time=tmax, verbose=False)
    kernel = hawkes.SimuHawkesExpKernels(adjacency=0.5,
                                         decays=1.6,
                                         baseline=[0.5])

    hawkes_1d.simulate()

    return

def make_3d_hawkes():
    from tick import hawkes
    adj_3d = np.array([[0.1, 0.1, 0.5],
                       [0.01, 0.2, 0.02],
                       [0.03, 0.0, 0.5]])
    decay_3d = 0.5
    baseline_3d = np.array([0.1, 0.15, 0.4])

    tmax = 180

    hawkes_3d = hawkes.SimuHawkesExpKernels(adj_3d,
                                            decay_3d,
                                            baseline=baseline_3d,
                                            end_time=tmax,
                                            verbose=False)

    hawkes_3d.reset()
    hawkes_3d.track_intensity(0.1)
    # hawkes_2d.simulate()

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


def make_syn_dataset(dim=2):
    if dim == 2:
        data = make_2d_hawkes()
    else:
        data = make_3d_hawkes()

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
    make_syn_dataset(dim=2)
