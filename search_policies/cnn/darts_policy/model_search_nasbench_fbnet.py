import numpy as np
import torch
import torch.nn as nn
import logging

from torch.distributions import Categorical, Gumbel

from ..operations import gumbel_softmax
from .model_search_nasbench import NASBENCH_CONFIG, NasBenchCellSearchDARTS, ModelSpec_DARTS,NasBenchNetSearchDarts


class NasBenchNetSearchFBNet(NasBenchNetSearchDarts):

    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearchDARTS):
        """
        Weight sharing nasbench version 1.

        :param input_channels:
        :param model_spec:
        :param config: defined in nasbench.config
        """
        input_channels = 3
        # wrapping to DARTS model spec
        model_spec = args.model_spec
        full_connection = 1 - np.tril(np.ones_like(model_spec.matrix))
        model_spec.matrix = full_connection
        matrix = full_connection
        self.num_intermediate_nodes = matrix.shape[0] - 2
        self.num_ops = 3  # TODO hardcode
        new_matrix = np.concatenate([np.zeros([1, self.num_intermediate_nodes]), matrix[0:-2, 1:-1]], axis=0) * 10
        alpha_topology = nn.Parameter(torch.tensor(new_matrix, dtype=torch.float32))
        alpha_ops = nn.Parameter(torch.tensor(np.zeros([self.num_ops, new_matrix.shape[0]]), dtype=torch.float32))
        model_spec = ModelSpec_DARTS(matrix, model_spec.ops)
        logging.info("Creating FBNet model spec: {}".format(model_spec))

        super(NasBenchNetSearchDarts, self).__init__(input_channels, model_spec, config, cell_cls, args)
        # this initialized by full rank list
        self.drop_path_keep_prob = 1 - self.args.drop_path_prob
        self.dropout = nn.Dropout(1 - self.args.keep_prob)
        self._criterion = nn.CrossEntropyLoss().cuda()
        # self._arch_parameters = None
        # this cause trouble, alpha topology and ops are not correctly duplicate to parallel.
        self.alpha_topology = alpha_topology
        self.alpha_ops = alpha_ops
        self._model_spec = model_spec
        self._model_spec.update_multi_gpu(self.alpha_topology, self.alpha_ops)
        self.backup_kwargs = {
            'args': args,
            'config': config,
            'cell_cls': cell_cls
        }
        self.gumbel_temperature = args.gumble_softmax_temp
        self.gumbel_temperature_decay = args.gumble_softmax_decay

        self.epoch = 0

    def temperature(self):
        # lower limit for temperature is 0.05, because too small can destroy training,
        # and this is to map with FB original 90 epochs setting
        return max(self.gumbel_temperature * pow(self.gumbel_temperature_decay, self.epoch), 0.05)

    def topology_weights(self, node):
        # return the soft weights for topology aggregation
        gumbel_dist = Gumbel(torch.tensor([.0]), torch.tensor([1.0]))
        return gumbel_softmax(self.alpha_topology[: node + 2, node], self.temperature(), gumbel_dist)[1:]

    def ops_weights(self, node):
        gumbel_dist = Gumbel(torch.tensor([.0]), torch.tensor([1.0]))
        return gumbel_softmax(self.alpha_ops[:, node], self.temperature(), gumbel_dist)

    def sample_model_spec(self, num):
        """
        Override, sample the alpha via gumbel softmax instead of normal softmax.
        :param num:
        :return:
        """
        alpha_topology = self.alpha_topology.detach().clone()
        alpha_ops = self.alpha_ops.detach().clone()
        sample_archs = []
        sample_ops = []
        gumbel_dist = Gumbel(torch.tensor([.0]), torch.tensor([1.0]))
        with torch.no_grad():
            for i in range(self.num_intermediate_nodes):
                # align with topoligy weights
                probs = gumbel_softmax(alpha_topology[: i+2, i], self.temperature(), gumbel_dist)
                sample_archs.append(Categorical(probs))
                probs_op = gumbel_softmax(alpha_ops[:, i], self.temperature(), gumbel_dist)
                sample_ops.append(Categorical(probs_op))

            return self._sample_model_spec(num, sample_archs, sample_ops)
