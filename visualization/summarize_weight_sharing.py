import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def summarize_weight_sharing_plot(gt_data, ws_data, gt_ylim=None, ws_ylim=None, filename='ws_vs_gt.pdf'):
    """

    :param gt_data: sorted.
    :param ws_data: index sort based on GT data.
    :param gt_ylim:
    :param ws_ylim:
    :param filename: save name.
    :return:
    """
    # plot the ground truth results
    avg_values, std_values = gt_data
    fig, ax = plt.subplots(figsize=(6, 5))

    # Indices to step through colormap
    map_name = 'GnBu'
    x = np.linspace(0.0, 1.0, 30 + len(avg_values) + 2)
    color_map = cm.get_cmap(map_name)
    rgb = color_map(x)[np.newaxis, :, :3]
    rgb = rgb[0][30: 30 + len(avg_values)]
    # print(rgb[0])
    # print(len(rgb))
    if gt_ylim:
        ax.set_ylim(gt_ylim)
    ax.spines['top'].set_visible(False)
    ax.bar(np.arange(0, len(avg_values)), avg_values, yerr=std_values, color=rgb,
           capsize=2)

    # Weight sharing data.
    ws_avg_values, ws_std_values = ws_data

    ax2 = ax.twinx()
    # ax2.set_ylabel('hehe', color='tab:blue')
    if ws_ylim:
        ax2.set_ylim(ws_ylim)
    ax2.tick_params('y', colors='red')
    ax2.errorbar(np.arange(0, len(ws_avg_values)), ws_avg_values,
                 yerr=ws_std_values, color='red',
                 alpha=0.6, fmt='-o',
                 capsize=2,
                 ecolor='black',
                 )
    ax2.spines['top'].set_visible(False)
    plt.legend(['Without WS', 'With WS'])
    # plt.grid(True)
    plt.savefig(filename)
    return plt
