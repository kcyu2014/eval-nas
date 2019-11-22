"""
2019.08.19, testing the first nasbench search space.
I should finish this within a weekend and should deploy this as soon as possible.
"""
from copy import copy
from functools import partial

import itertools
import logging
import os
from collections import deque, OrderedDict

import IPython

from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import NASBench_v2, ModelSpec_v2
from search_policies.cnn.search_space.nas_bench.sampler import obtain_full_model_spec
from search_policies.cnn.search_space.nas_bench.util import change_model_spec
from ..api import CNNSearchSpace
import numpy as np
from .model import NasBenchNet


class NASbenchSearchSpace(CNNSearchSpace):
    sample_step_for_evaluation = 2
    top_K_complete_evaluate = 200
    evaluate_ids = None             # stores the pool of evaluate ids. Fixed after initialized.
    evaluate_model_spec_ids = None  # Current pool of ids, can change from time to time.

    def __init__(self, args, full_dataset=False):
        """
        Transferring from original.
        :param args:
        """
        # self.args = args
        super(NASbenchSearchSpace, self).__init__(args)
        self.topology_fn = NasBenchNet
        self.sample_step_for_evaluation = 2 if not args.debug else 30
        # read nasbench related configs.
        args.model_spec = obtain_full_model_spec(args.num_intermediate_nodes + 2)
        v = self.args.num_intermediate_nodes + 2
        self.nasbench = NASBench_v2(os.path.join(self.args.data, 'nasbench/nasbench_only108.tfrecord'),
                                    config=f'v{v}_e9_op3', only_hash=not full_dataset)
        self.nasbench_hashs, self.nasbench_model_specs = self.nasbench.model_hash_rank(full_spec=True)
        self.rank_by_mid = [i for i in range(0, len(self.nasbench_hashs))]
        self.available_ops = self.nasbench.available_ops
        self.initialize_evaluate_pool()

    ## This belongs to interaction for now, should be removed later.
    @property
    def topologies(self):
        return self.nasbench_model_specs

    @property
    def hashs(self):
        return self.nasbench_hashs

    @property
    def num_architectures(self):
        return len(self.nasbench_hashs)

    def initialize_evaluate_pool(self):
        # Process evaluation nodes
        self.evaluate_ids = [i for i in range(0, self.num_architectures, self.sample_step_for_evaluation)]
        # remove the landmark id from eval_ids

        self.evaluate_model_spec_ids = deque(self.evaluate_ids)
        if len(self.evaluate_model_spec_ids) > self.top_K_complete_evaluate:
            self.evaluate_model_spec_ids = deque(sorted(
                np.random.choice(self.evaluate_model_spec_ids, self.top_K_complete_evaluate, replace=False).tolist()))

    def evaluate_model_spec_id_pool(self):
        if len(self.evaluate_model_spec_ids) > self.top_K_complete_evaluate:
            self.evaluate_model_spec_ids = deque(sorted(
                np.random.choice(self.evaluate_model_spec_ids, self.top_K_complete_evaluate,
                                 replace=True).tolist()))
        return self.evaluate_model_spec_ids

    def eval_model_spec_id_append(self, mid):
        if mid in self.evaluate_model_spec_ids:
            self.evaluate_model_spec_ids.remove(mid)

        if len(self.evaluate_model_spec_ids) >= self.top_K_complete_evaluate:
            old_arch = self.evaluate_model_spec_ids.pop()
            logging.debug("Pop arch {} from pool".format(old_arch))
        self.evaluate_model_spec_ids.append(mid)

    def eval_model_spec_id_rank(self, ids, perfs):
        """
        Rank the evaluate id pools by the performance.

        :param ids:
        :param perfs:
        :return: None
        """
        # rank the thing, make sure the pop-left will eliminate the poor performed archs.
        old_archs_sorted_indices = np.argsort(perfs)[::-1]
        rank_ids = [ids[i] for i in old_archs_sorted_indices]
        if len(rank_ids) > self.top_K_complete_evaluate:
            rank_ids = rank_ids[:self.top_K_complete_evaluate]
        self.evaluate_model_spec_ids = deque(rank_ids)

    def random_topology(self):
        """
        Naive random sampling method.
        :return: id, spec
        """
        rand_spec_id = self.random_ids(1)[0]
        rand_spec = self.nasbench_model_specs[rand_spec_id]
        return rand_spec_id, rand_spec

    def random_ids(self, number):
        return sorted(np.random.choice(np.arange(0, self.num_architectures), number, replace=False).tolist())

    def random_eval_ids(self, number):
        """
        Random a couple of eval ids from the Evaluation pool, but not the current ids.
        :param number:
        :return:
        """
        return sorted(np.random.choice(self.evaluate_ids, min(number, len(self.evaluate_ids)),
                                       replace=False).tolist())

    def hash_by_id(self, i):
        """ return model hash by id """
        return self.nasbench_hashs[i]

    def topology_by_id(self, i):
        """ return topoligy by id """
        return self.nasbench_model_specs[i]

    def validate_model_indices(self, valid_queue_length, sampling=None):
        """
        Process for validation step during training supernet.
        This step for the current being is a random selection without any prior knowledge.
        Possible to support another version.

        :param valid_queue_length:
        :return: valid_model_pool
        """
        nb_models = self.num_architectures
        nb_batch_per_model = max(valid_queue_length // nb_models, 1)
        if sampling is None:
            valid_model_order = np.random.choice(range(nb_models), nb_models, False)
        else:
            raise NotImplementedError("not yet supported. to add in future.")

        if nb_models > valid_queue_length:
            valid_model_order = valid_model_order[:valid_queue_length]
            nb_models = valid_queue_length
        return nb_batch_per_model, nb_models, valid_model_order

    def replace_eval_ids_by_random(self, number):
        """ Random a subset and replace the bottom performed architectures. """
        replace_number = 0
        rand_eval_ids = self.random_eval_ids(number)
        for eid in rand_eval_ids:
            if eid not in self.evaluate_model_spec_ids:
                self.eval_model_spec_id_append(eid)
                replace_number += 1
        return replace_number

    def process_archname_by_id(self, arch):
        # arch is mid
        return f"{arch}, {self.hashs[arch]}"

    def generate_new_arch(self, number):
        """
        Return the id or not.
        :param number:
        :return:
        """
        archs = []
        for _ in range(number):
            _, m = self.random_topology()
            archs.append(m)
        return archs

    # for sparse kendall tau
    def query_gt_perfs(self, model_ids):
        """
        return the testing accuracy.
        :param model_ids: ids for the given model
        :return: gt performance of this.
        """

        return [self.nasbench.perf_rank[i][1] for i in model_ids]


class NasBenchSearchSpaceLinear(NASbenchSearchSpace):

    def __init__(self, args):
        super(NasBenchSearchSpaceLinear, self).__init__(args)
        # process only linear labels
        self.original_model_specs = self.nasbench_model_specs
        self.original_hashs = self.nasbench_hashs
        self.sample_step_for_evaluation = 1
        self.process_nasbench_linear()

    def process_nasbench_linear(self):
        """ Process nasbench linear search space. This is a much simpler search space. """
        # only take the linear architectures. a much simpler space.
        full_spec = obtain_full_model_spec(self.args.num_intermediate_nodes)
        matrix = np.eye(self.args.num_intermediate_nodes + 2, self.args.num_intermediate_nodes + 2, 1).astype(np.int)
        # indeed, we need also add ranks.
        self.nasbench_hashs = []
        self.nasbench_model_specs = []
        specs = OrderedDict()
        hashs = OrderedDict()
        for labeling in itertools.product(*[range(len(self.nasbench.available_ops))
                                            for _ in range(self.args.num_intermediate_nodes)]):
            ops = ['input', ] + [self.nasbench.available_ops[i] for i in labeling] + ['output',]
            new_spec = ModelSpec_v2(matrix.copy(), copy(ops))
            new_hash = new_spec.hash_spec()
            _id = self.original_hashs.index(new_hash)
            specs[_id] = new_spec
            hashs[_id] = new_hash

        rank_key = sorted(hashs.keys())
        self.nasbench_hashs = [hashs[_id] for _id in rank_key]
        self.nasbench_model_specs = [specs[_id] for _id in rank_key]
        self.sample_step_for_evaluation = 1
        self.initialize_evaluate_pool()
        # IPython.embed(header='check this is correct or not')
        logging.info("Linear space, totoal architecture number is {}".format(self.num_architectures))

    def evaluate_model_spec_id_pool(self):
        return self.evaluate_model_spec_ids


class NasBenchSearchSpaceSubsample(NASbenchSearchSpace):

    # keep track of original space ids, because new id will be flushed.
    rank_id_in_original_nasbench = []

    def __init__(self, args):
        super(NasBenchSearchSpaceSubsample, self).__init__(args)
        self.original_model_specs = self.nasbench_model_specs
        self.original_hashs = self.nasbench_hashs
        self.sample_step_for_evaluation = 1
        self.process_subsample_space()

    def process_subsample_space(self):
        # raise NotImplementedError('finish later')
        sample_num = min(self.args.num_archs_subspace, self.num_architectures)
        subspace_ids = sorted([int(a) for a in np.random.choice(
            len(self.nasbench_model_specs), sample_num, replace=False)])
        self.rank_id_in_original_nasbench = subspace_ids
        self.nasbench_hashs = [self.original_hashs[_id] for _id in subspace_ids]
        self.nasbench_model_specs = [self.original_model_specs[_id] for _id in subspace_ids]
        self.initialize_evaluate_pool()
        print("Random subspace with {} architectures: {}".format(self.num_architectures, subspace_ids[:100]))
        print("Evaluation architecture pool: {}".format(self.evaluate_model_spec_ids))


def nodes_to_key(nodes):
    # always [0, 1, 2, ..., num_intermediate_node]
    # nodes = range(len(nodes))
    # return ','.join(map(str, nodes))
    return len(nodes)

def key_to_nodes(key):
    # return [int(a) for a in key.split(',')]
    return list(range(key))

def model_spec_to_involving_nodes(spec):
    matrix = spec.matrix.copy()
    active_nodes = np.argwhere(matrix.sum(axis=1)[1:-1] > 0).reshape(-1)
    return active_nodes.tolist(), matrix


def permunate_ops_all(n, OPS):
    if n == 0:
        yield []
    elif n == 1:
        for o in OPS:
            yield [o,]
    else:
        for o in OPS:
            for rest_ops in permunate_ops_all(n-1, OPS):
                yield [o,] + rest_ops


def permunate_ops_last_node(n, OPS, default_pos=0):
    for o in OPS:
        yield [OPS[default_pos], ] * (n-1) + [o,]


def permutate_ops_given_topology(matrix, OPS, permutate_last=True):
    # print('permutate under topology matrix ', matrix)
    node = matrix.shape[0] - 2
    if permutate_last:
        all_ops = permunate_ops_last_node(node, OPS, default_pos=0)
    else:
        all_ops = permunate_ops_all(node, OPS)
    for ops in all_ops:
        ops = ['input', ] + ops + ['output',]
        copy_matrix = matrix.copy()
        a = ModelSpec_v2(copy_matrix, ops)
        if a.valid_spec:
            yield a


class NasBenchSearchSpaceFairNasTopology(NASbenchSearchSpace):

    def __init__(self, args):
        super(NasBenchSearchSpaceFairNasTopology, self).__init__(args)
        self.nasbench_involving_nodes = OrderedDict()
        for ind, spec in enumerate(self.topologies):
            active_nodes, matrix = model_spec_to_involving_nodes(spec)
            key = nodes_to_key(active_nodes)
            if key in self.nasbench_involving_nodes.keys():
                self.nasbench_involving_nodes[key].append(ind)
            else:
                self.nasbench_involving_nodes[key] = [ind, ]
        self.nasbench_topo_sample_probs = []
        for k, v in self.nasbench_involving_nodes.items():
            logging.debug(f'involving nodes {k} : num arch {len(v)}')
            self.nasbench_topo_sample_probs.append(len(v))
        self.nasbench_topo_sample_probs = list(reversed(self.nasbench_topo_sample_probs))

    def nasbench_sample_matrix_from_list(self, nodes, probs):
        """
        Recursively sample from the list of data by a prob.
        This cooperates with fairnas topology sampler.
        Fair sampling.

        :param nodes: [1, ... interm,] node id as a list
        :param probs: probability to sample an list with length equal to probs, len(probs) == len(data)
        :return:
        """

        def normalize(probs):
            return list(map(lambda x: float(x / sum(probs)), probs))

        if len(nodes) == 0:
            return [None,]
        else:
            try:
                total = self.args.num_intermediate_nodes
                probs = normalize(probs)
                num_sample = np.random.choice(np.arange(len(nodes) + 1), 1, p=probs)
                sample_nodes = sorted(np.random.choice(nodes, num_sample, replace=False))
                rest_nodes = list(set(nodes) - set(sample_nodes))
                new_probs = probs[:len(rest_nodes) + 1]
                # nasbench matrix including input and output.
                topo_matrices_ids = self.nasbench_involving_nodes[nodes_to_key(sample_nodes)]
                sample_id = np.random.choice(topo_matrices_ids, 1)[0]
                sample_matrix = self.topologies[sample_id].matrix.copy()
                if sample_matrix.shape[0] == total + 2:
                    # terminate the looping.
                    return [sample_matrix, None]
                else:
                    # Make sample nodes to full matrix spec.
                    sample_nodes = [0,] + sample_nodes + [total + 1]
                    # make new_matrix[sample_nodes,:][:, sample_nodes] = sample_matrix
                    matrix = np.zeros([total + 2, total + 2], dtype=int)
                    _matrix = matrix[sample_nodes,:]
                    _matrix[:, sample_nodes] = sample_matrix
                    matrix[sample_nodes,:] = _matrix
                    return [matrix,] + self.nasbench_sample_matrix_from_list(rest_nodes, new_probs)
            except Exception as e:
                logging.error(f'{e}')
                IPython.embed(header='Check mistake of nasbench_sample_matrix_from_list')


class NasBenchSearchSpaceICLRInfluenceWS(NasBenchSearchSpaceSubsample):
    arch_hash_by_group = {}
    arch_ids_by_group = {}

    # preprocess this search space.
    def process_subsample_space(self):
        # composing the linear search space.
        # Variates of ops, but topology is sampled from a pool.
        nodes = self.args.num_intermediate_nodes
        AVAILABLE_OPS = self.nasbench.available_ops
        logging.info("Processing NASBench WS influence Search Space ...")
        permutate_op_fn = partial(permutate_ops_given_topology,
                                  permutate_last=True)
        logging.info("Permutating the last node only? {}".format(
            not self.args.nasbench_search_space_ws_influence_full))

        subspace_ids = []
        subspace_model_specs_dict = {}
        # make all possible matrix:
        for i in range(nodes):
            matrix = np.zeros((nodes + 2, nodes + 2), dtype=np.int)
            matrix[nodes, -1] = 1  # connect output to node n-1.
            matrix[i, -2] = 1  # connect last node to one of the previous node.
            if i > 0:
                if i > 1:
                    matrix[0:i, 1:i + 1] = np.triu(np.ones((i, i), dtype=np.int))
                else:
                    matrix[0, 1] = 1

            logging.info(f'Node {i}-{nodes} connection: {matrix}')

            self.arch_hash_by_group[i] = []
            self.arch_ids_by_group[i] = []

            for spec in permutate_op_fn(matrix, AVAILABLE_OPS):
                hash = spec.hash_spec()
                spec.resume_original()
                try:
                    _id = self.nasbench_hashs.index(hash)
                except ValueError as e:
                    logging.error("Spec is not valid here: {}".format(e))
                    logging.error(spec)
                    continue
                # subspace_ids.append(_id)
                if hash not in subspace_model_specs_dict.keys():
                    # only keep one spec.
                    subspace_model_specs_dict[hash] = spec
                    self.arch_hash_by_group[i].append(hash)
                    self.arch_ids_by_group[i].append(_id)
                    subspace_ids.append(_id)

        # count = 0
        # for i in range(nodes):
        #     n_g = []
        #     n_id = []
        #     # removing the duplicated items
        #     logging.info(f"Process rank group {i}, original length {len(self.arch_ids_by_group[i])} ... ")
        #     for _id, h in zip(self.arch_ids_by_group[i], self.arch_hash_by_group):
        #         if _id not in n_id:
        #             n_id.append(_id)
        #             n_g.append(h)
        #     self.arch_ids_by_group[i] = n_id
        #     self.arch_hash_by_group[i] = n_g
        #     assert len(n_id) == len(n_g)
        #     count += len(n_id)
        #     logging.info("Length after processing: {}".format(self.arch_ids_by_group[i]))

        sort_ids = np.argsort(subspace_ids)
        sort_subspace_ids = [subspace_ids[i] for i in sort_ids]
        self.nasbench_model_specs_prune = [self.original_model_specs[i] for i in sort_subspace_ids]
        self.nasbench_hashs = [self.original_hashs[_id] for _id in sort_subspace_ids]
        self.nasbench_model_specs = [subspace_model_specs_dict[h] for h in self.nasbench_hashs]
        self.initialize_evaluate_pool()
        logging.info("Totally {} architectures: {}".format(self.num_architectures, subspace_ids[:100]))
        logging.info("Evaluation architecture pool: {}".format(self.evaluate_model_spec_ids))
