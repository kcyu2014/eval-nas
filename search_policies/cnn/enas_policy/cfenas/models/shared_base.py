import numpy as np
import torch

from models.wpl_module import WPLModule


def size(p):
    return np.prod(p.size())


class SharedModel(WPLModule):

    def __init__(self, args=None):
        super(SharedModel, self).__init__(args=args)

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def init_training(self, batch_size):
        return None

    def get_f(self, name):
        """

        :param name:
        :return:
        """
        raise NotImplementedError()

    def get_num_cell_parameters(self, dag):
        raise NotImplementedError()

    def reset_parameters(self):
        raise NotImplementedError()

