"""
Implement the FairNAS in a topology fair way.

"""
import logging
import os
from collections import OrderedDict
from functools import partial

import IPython
import numpy as np
from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import ModelSpec_v2
from search_policies.cnn.search_space.nas_bench.nasbench_search_space import NasBenchSearchSpaceFairNasTopology
from search_policies.cnn.search_space.nas_bench.sampler import obtain_full_model_spec
from search_policies.cnn.procedures.fairnas_procedure import fairnas_train_model_v2
from search_policies.cnn.random_policy.nasbench_weight_sharing_policy import NasBenchNetOneShotPolicy
from search_policies.cnn.search_space.nas_bench.util import change_model_spec


class NasBenchNetTopoFairNASPolicy(NasBenchNetOneShotPolicy):

    def __init__(self, args):
        super(NasBenchNetTopoFairNASPolicy, self).__init__(args)
        self.search_space = NasBenchSearchSpaceFairNasTopology(args)
        self.train_fn = partial(fairnas_train_model_v2,
                                args=self.args, architect=None,
                                topology_sampler=self.random_sampler,
                                op_sampler=self.op_sampler
                                )

    def random_sampler(self, model, architect, args):
        # according to Aug 8 meeting. become the new topo sampler
        total = self.args.num_intermediate_nodes
        matrices_list = self.search_space.nasbench_sample_matrix_from_list(
            np.arange(1, total+1), self.search_space.nasbench_topo_sample_probs)

        for matrix in matrices_list:
            if matrix is not None:
                spec = obtain_full_model_spec(total + 2)
                try:
                    spec = ModelSpec_v2(matrix, spec.ops)
                except:
                    IPython.embed()
                self.model_spec = spec
                self.model_spec_id = None
                yield change_model_spec(model, spec)

    def op_sampler(self, model, architect, args):
        return super(NasBenchNetTopoFairNASPolicy, self).op_sampler(model, architect, args)
