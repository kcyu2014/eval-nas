# TO unite NASBench and the DARTS / NAO and ENAS based search space...
# modeling the DARTS or NAO sampler into NASBench, not the other way around. it is impossible.
import collections
import json
import sys

import itertools
import logging

from search_policies.cnn.darts_policy.genotypes import Genotype
from search_policies.cnn.darts_policy.operations import OPS as DARTS_OPS
from search_policies.cnn.nao_policy.operations import OPERATIONS as NAO_SMALL_OPS
from search_policies.cnn.nao_policy.operations import OPERATIONS_large as NAO_OPS
from search_policies.cnn.search_space.nas_bench.genotype import ALLOWED_OPS as NASBENCH_OPS
from search_policies.cnn.search_space.nas_bench.genotype import model_spec_demo, test_model_max
from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import NASBench_v2

import numpy as np
from nasbench.lib import graph_util


def nasbench_hash_to_model_spec(hash):
    if hash == 'random':
        model_spec = model_spec_demo
    elif hash == 'test-1':
        model_spec = test_model_max(1)
    else:
        # Read as 'hash'
        logging.info("Treat the arch as hash: {}".format(hash))
        nasbench_v2 = NASBench_v2('./data/nasbench/nasbench_only108.tfrecord', only_hash=True)
        logging.info("Load from NASBench dataset v2")
        model_spec = nasbench_v2.hash_to_model_spec(hash)
    return model_spec


def nasbench_search_space(args, file_location=None):
    FLAGS = args
    total_graphs = 0    # Total number of graphs (including isomorphisms)
    # hash --> (matrix, label) for the canonical graph associated with each hash
    buckets = {}

    logging.info('Using %d vertices, %d op labels, max %d edges',
                 FLAGS.max_vertices, FLAGS.num_ops, FLAGS.max_edges)
    for vertices in range(2, FLAGS.max_vertices+1):
        for bits in range(2 ** (vertices * (vertices-1) // 2)):
            # Construct adj matrix from bit string
            matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
                                     (vertices, vertices),
                                     dtype=np.int8)

            # Discard any graphs which can be pruned or exceed constraints
            if (not graph_util.is_full_dag(matrix) or
                    graph_util.num_edges(matrix) > FLAGS.max_edges):
                continue

            # Iterate through all possible labelings
            for labeling in itertools.product(*[range(FLAGS.num_ops)
                                                for _ in range(vertices-2)]):
                total_graphs += 1
                labeling = [-1] + list(labeling) + [-2]
                fingerprint = graph_util.hash_module(matrix, labeling)

                if fingerprint not in buckets:
                    buckets[fingerprint] = (matrix.tolist(), labeling)

                # This catches the "false positive" case of two models which are not
                # isomorphic hashing to the same bucket.
                elif FLAGS.verify_isomorphism:
                    canonical_graph = buckets[fingerprint]
                    if not graph_util.is_isomorphic(
                            (matrix.tolist(), labeling), canonical_graph):
                        logging.fatal('Matrix:\n%s\nLabel: %s\nis not isomorphic to'
                                      ' canonical matrix:\n%s\nLabel: %s',
                                      str(matrix), str(labeling),
                                      str(canonical_graph[0]),
                                      str(canonical_graph[1]))
                        sys.exit()

        logging.info('Up to %d vertices: %d graphs (%d without hashing)',
                     vertices, len(buckets), total_graphs)

    with open(file_location, 'w') as f:
        json.dump(buckets, f, sort_keys=True)


def sequence_arch_to_model_spec(arch, ops=NASBENCH_OPS):
    # arch = [0, 1, 1, 2, 2, 3,]
    # num_layer = int(len(arch) / 2)
    # is_output = [True,] * (num_layer + 1)
    # matrix = np.zeros((num_layer + 2, num_layer + 2),dtype=np.int)
    # opers = []
    # for i in range(1, num_layer + 1):
    #     matrix[i][arch[(i-1)*2]] = 1
    #     is_output[arch[(i-1)*2]] = False
    #     opers.append(ops[arch[i*2 + 1]])

    return model_spec_demo


def genotype_to_model_spec(genotype, ops=NASBENCH_OPS):
    # genotype =
    pass

def genotype_to_networkx_dag(genotype):
    """

    Transform the genotype into DAG format.
    DAG can be in networkx, so to directly search in the NASBench 101.

    :param genotype: (normal and reduce is the same)
    :return:
    """
    # assert genotype.reduced == genotype.normal, 'For this transformation, reduced must match '
    # ignore the node id 0 in genotypes.
    # genotype is the edge list accordingly.

    # create the edge list
    def _create_edge_list(genotype, concat):
        ops = ['prev_input', 'input']
        edge_list = []
        assert len(genotype) % 2 == 0
        intermediate_node = int(len(genotype) / 2)
        for i in range(intermediate_node):
            curr_idx = i + 1
            if genotype[i][0] == 0:
                raise ValueError("Cannot connect 0 because previous input is not valid in NASBench!")
            ops.append(genotype[i][0])
            edge_list.append((genotype[i][1], curr_idx, {'weight': 1}))
        output_id = intermediate_node + 2
        # add output
        for i in concat:
            pass
    raise NotImplementedError("This is not possible, to think about something more...")


class CNNSearchSpace:
    def __init__(self, args):
        self.args = args
        self.intermediate_nodes = self.args.num_intermediate_nodes
        self.num_operations = None
        self.num_solutions = None
        self.operations = None

    def genotype_id_from_genotype(self, genotype):
        raise NotImplementedError("override")

    def genotype_from_genotype_id(self, genotype_id):
        raise NotImplementedError("override")

    def sample_architectures(self, n=1):
        raise NotImplementedError


class PreAddSearchSpace:
    """ Pre addition is the search space adopted by NASBench.
        Core idea is, operation lies on node
            A ----|
                  |-> + --> Op ---> output
            B ----|

        This should be a bridge between random weight sharing training v.s.

        Version 1. To match up the genotype of NASBenchNet.
        Formality of genotype and genotype_id is similar to RNN case.

        Purpose is just a
    """
    def __init__(self, args):
        self.args = args

    def geno_to_model_spec(self):
        pass


class PostAddSearchSpace(CNNSearchSpace):
    """ Post addition is the search space adopted by DARTS/NAO and so on ...
            Core idea is, operation lies on Edge!
                A --> Op1 --|
                            |-> + -> output
                B --> Op2 --|
        """

    def genotype_from_genotype_id(self, genotype_id):
        pass

    def genotype_id_from_genotype(self, genotype):
        pass

    def to_darts_genotype(self, genotype):
        # must be nao genotype. list of string.
        pass

    def to_nao_genotype(self, genotype):
        # must be darts genotype, Genotype
        pass


#### For the search_main.py to retrieve some architecture #####
def get_fixed_model_spec(args):
    """
    Getting the fixed model spec according to various settings.
    :param args:
    :return:
    """
    model_spec = None
    if args.searc_space.lower() == 'nasbench':
        model_spec = nasbench_hash_to_model_spec(args.model_spec_hash)
    elif args.searc_space.lower() in ['darts', 'nao']:
        raise NotImplementedError
    else:
        raise ValueError("Search space not supported ", args.searc_space.lower())
    return model_spec
