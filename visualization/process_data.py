# Processing the log.txt
import numpy as np


def tensorboard_summarize_list(list_data, writer, key, step,
                               top_k=(1,5), std=True,
                               ascending=True
                               ):
    """

    :param list_data:
    :param writer:
    :param key:
    :param step:
    :param top_k:
    :param std:
    :param ascending: True for loss, False for Accuracy.
    :return:
    """

    list_data = sorted(list_data)
    if not ascending:
        list_data.reverse()
    value_pairs = {}
    np_d = np.asanyarray(list_data, dtype=np.float32)
    value_pairs[key + '/mean'] = np_d.mean()
    if std:
        value_pairs[key + '/std'] = np_d.std()
    if top_k:
        for i in top_k:
            value_pairs[key + '/top-{}'.format(i)] = np_d[:i].mean()
    for k, v in value_pairs.items():
        writer.add_scalar(k, v, step)
    return writer


def txt_to_dict(log='log.txt'):
    with open(log, 'r') as f:
        lines = f.readlines()

    # keyword : end of epoch
    epoch_infos = [l[:-1] for l in lines if l.find('end of epoch') > 0]

