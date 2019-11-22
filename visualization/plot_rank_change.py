
# Ploting the Rank change data
from operator import itemgetter

import IPython
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
import pandas as pd

import utils
from visualization.best_configs import BEST_RANK_BY_GENOTYPE


def get_cmap_as_rgbs(nb_points=32):
    cmaps = OrderedDict()
    map_name = 'YlGnBu'
    # Indices to step through colormap
    x = np.linspace(0.0, 1.0, nb_points + 30)
    color_map = cm.get_cmap(map_name)
    rgb = color_map(x)[np.newaxis, :, :3]
    rgb = rgb[0][15:nb_points + 15]
    rgb = [rgb[len(rgb) - 1 - i] for i in range(len(rgb))]
    return rgb


# Plot code
# Rank change without weight sharing.
# print(rank_change_data)

def process_rank_data_nasbench(data, filename):
    if 'ranking_per_epoch' in data.keys():
        ranking_per_epoch = data['ranking_per_epoch']
    else:
        ranking_per_epoch = data

    epochs = ranking_per_epoch.keys()

    # obtain the ground-truth data from 'Rank'.
    _d = next(iter(ranking_per_epoch.values()))
    p = sorted([elem[1] for elem in _d], key=itemgetter(3))
    rgb = get_cmap_as_rgbs(len(p))
    rank_gt = np.array([elem.geno_id for elem in p], dtype=np.uint)
    # Geno id is the gt-rank. so
    # reverset the gt_index_by_rank, to check if the results make more sence later.
    gt_index_by_rank = np.arange(len(p)-1, -1, -1)
    e_dict = {
        'geno_id': [],
        'genotype': [],
    }
    for ind, i in enumerate(epochs):
        epoch_data = ranking_per_epoch[i]
        epoch_data = sorted(epoch_data, key=lambda y: y[1][0])

        e_dict['epoch_{}'.format(i)] = []

        for geno_hash, (valid_acc, _, geno_id, _) in epoch_data:
            if ind == 0:
                e_dict['genotype'].append(geno_hash)
                e_dict['geno_id'].append(geno_id)
            e_dict['epoch_{}'.format(i)].append(valid_acc)

    rc_data, x_values, rank_df = _process_rank_data_for_plotting(ranking_per_epoch, gt_index_by_rank, e_dict)
    return plot_rank_change(rc_data, x_values, y_labels=rank_df['genotype'].values,
                            rgb=rgb,
                            show=False,
                            save_path=filename)


def _process_rank_data_for_plotting(rank_data, gt_index_by_rank, e_dict):
    """
    :param rank_data:
    :param gt_index_by_rank:
    :param e_dict: To create panda data. {}
    :return:
    """
    epochs = rank_data.keys()

    # print(e_dict)
    # IPython.embed()
    pd_epoch = pd.DataFrame.from_dict(e_dict)
    pd_epoch.sort_values('geno_id', inplace=True)
    pd_epoch.reset_index(inplace=True, drop=True)
    pd_epoch = pd_epoch.reindex(gt_index_by_rank)
    pd_epoch.reset_index(inplace=True, drop=True)
    # print(pd_epoch)

    # rank change
    rank_df = pd_epoch
    rc_data = []  # rc = rank change
    keys_list = list(epochs)
    step = 2
    x_values = []
    for i in range(0, len(keys_list), step):
        k = keys_list[i]
        a = rank_df.sort_values(['epoch_{}'.format(k)])
        x_values.append(int(k))
        rc_data.append(a.index.values)
    # Add the final result.
    rc_data.append([i for i in range(0, len(rc_data[-1]))])
    x_values.append(x_values[-1] + 1 * (x_values[-1] - x_values[-2]) if len(x_values) > 1 else 1)

    x_values = np.array(x_values)
    rc_data = np.asanyarray(rc_data)
    rc_data = np.transpose(rc_data)
    return rc_data, x_values, rank_df


def process_rank_data_node_2(data, filename):
    """
    Rank data post-processing. It is specialized for the data saving with
    rank_gens f
    :param filename:
    :return:
    """
    epochs = data['ranking_per_epoch'].keys()
    e_dict = {
        'geno_id': [],
        'genotype': [],
    }
    for ind, i in enumerate(epochs):
        epoch_data = data['ranking_per_epoch'][i]
        epoch_data = sorted(epoch_data, key=lambda y: y[1][1])

        e_dict['epoch_{}'.format(i)] = []
        for geno, (valid_ppl, geno_id) in epoch_data:
            if ind == 0:
                e_dict['genotype'].append(geno)
                e_dict['geno_id'].append(geno_id)
            e_dict['epoch_{}'.format(i)].append(valid_ppl)

    # print(e_dict)
    pd_epoch = pd.DataFrame.from_dict(e_dict)
    pd_epoch.sort_values('geno_id', inplace=True)
    pd_epoch.reset_index(inplace=True, drop=True)
    pd_epoch = pd_epoch.reindex(BEST_RANK_BY_GENOTYPE)
    pd_epoch.reset_index(inplace=True, drop=True)
    # print(pd_epoch)

    # rank change
    rank_df = pd_epoch
    rc_data = []  # rc = rank chan
    keys_list = list(epochs)
    step = 2
    x_values = []
    for i in range(0, len(keys_list), step):
        k = keys_list[i]
        a = rank_df.sort_values(['epoch_{}'.format(k)])
        x_values.append(int(k))
        rc_data.append(a.index.values)
    # Add the final result.
    rc_data.append([i for i in range(0, len(rc_data[-1]))])
    x_values.append(x_values[-1] + 1 * (x_values[-1] - x_values[-2]) if len(x_values) > 1 else 1)

    x_values = np.array(x_values)
    rc_data = np.asanyarray(rc_data)
    rc_data = np.transpose(rc_data)

    # print(rc_data)
    # print(rc_data.shape)
    # print(rank_df['genotype'].values)
    # print(rank_df['geno_id'].values)
    # Return the figure object for tb to add
    return plot_rank_change(rc_data, x_values,
                            y_labels=rank_df['genotype'].values,
                            rgb=get_cmap_as_rgbs(32),
                            show=False, save_path=filename)


def plot_rank_change(rank_change_data,
                     x_values,
                     y_labels,
                     rgb,
                     show=False, save_path='rank-change.pdf',
                     ):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    y_dim, x_dim = rank_change_data.shape
    for i in range(y_dim):
        ax.plot(x_values, rank_change_data[i], linestyle='--', linewidth=1, color=rgb[i], label=y_labels[i])
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(i) for i in x_values[:-1]] + ['final'])
    ax.legend(bbox_to_anchor=(1.3, 0.5), loc='center')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig

