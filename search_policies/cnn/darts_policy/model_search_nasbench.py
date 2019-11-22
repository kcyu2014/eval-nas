"""
Nasbench adaptation for NAO.
"""

import logging
from operator import itemgetter

import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import IPython
from torch.distributions import Categorical

from search_policies.cnn.search_space.nas_bench.model_search import NasBenchCellSearch
from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import ModelSpec_v2
from search_policies.cnn.search_space.nas_bench.model_builder import compute_vertex_channels
from nasbench.lib import config as _config
from search_policies.cnn.search_space.nas_bench.operations import conv_bn_relu, MixedVertexDARTS
from search_policies.cnn.search_space.nas_bench.model import NasBenchNet, NasBenchCell
import torch.nn.functional as F

NASBENCH_CONFIG = _config.build_config()


class ModelSpec_DARTS(ModelSpec_v2):
    # This is a dynamic model spec for NasBenchNetSearchDarts.

    # n = intermediate nodes.
    alpha_topology = None
    AVAILABLE_OPS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
    # This is size [n + 1, n], [:-1, :] is as original matrix without input and output [1:-1, 1:-1].
    # [-1, :] is used for zero operation
    # to extract operation, use softmax([:, node_id]).
    # m = num_ops
    alpha_ops = None
    # size [m, n]. sample m operations accordingly.
    # Extract operation: softmax([:, node_id]).

    def __init__(self, matrix, ops, channel_last=True):
        super(ModelSpec_DARTS, self).__init__(matrix, ops, channel_last)
        self.num_intermediate_nodes = matrix.shape[0] - 2
        self.num_ops = 3  # TODO hardcode
        self.alpha_ops = None
        self.alpha_topology = None

    def update_multi_gpu(self, alpha_topology, alpha_ops):
        self.alpha_topology, self.alpha_ops = alpha_topology, alpha_ops

    def weights(self, node):
        """Sample weights for input node"""
        return self.topology_weights(node), self.ops_weights(node)

    def topology_weights(self, node):
        # return the soft weights for topology aggregation
        return nn.functional.softmax(self.alpha_topology[: node + 2, node])[1:]

    def ops_weights(self, node):
        return nn.functional.softmax(self.alpha_ops[:, node])

    def sample_model_spec(self, num):
        new_model_specs = []
        alpha_topology = self.alpha_topology.detach().clone()
        alpha_ops = self.alpha_ops.detach().clone()
        sample_archs = []
        sample_ops = []
        with torch.no_grad():
            for i in range(self.num_intermediate_nodes):
                # align with topoligy weights
                probs = nn.functional.softmax(alpha_topology[: i+2, i])
                sample_archs.append(Categorical(probs))
                probs_op = nn.functional.softmax(alpha_ops[:, i])
                sample_ops.append(Categorical(probs_op))
            for _ in range(num):
                new_matrix = np.zeros((self.num_intermediate_nodes + 2,self.num_intermediate_nodes + 2), dtype=np.int)
                new_ops = ['input',] + [None,] * self.num_intermediate_nodes + ['output']
                for i in range(self.num_intermediate_nodes):
                    # action = 0 means, sample drop path
                    action = sample_archs[i].sample() - 1
                    if -1 < action < i + 1: # cannot sample current node id
                        new_matrix[action, i + 1] = 1
                    # sample ops
                    op_action = sample_ops[i].sample()
                    new_ops[i + 1] = self.AVAILABLE_OPS[op_action]
                # logging.debug("Sampled architecture: matrix {}, ops {}".format(new_matrix, new_ops))
                new_matrix[:, -1] = 1 # connect all output
                new_matrix[-1, -1] = 0 # make it upper trianguler
                mspec = ModelSpec_v2(new_matrix, new_ops)
                # logging.debug('Sampled model spec {}'.format(mspec))
                new_model_specs.append(mspec)
                # mspec.hash_spec(self.AVAILABLE_OPS)
        return new_model_specs

    def summary(self, num_sample):
        # This is to sample num_archs times, rank them by the number.
        count = {}
        model_specs = self.sample_model_spec(num_sample)

        for m in model_specs:
            h = m.hash_spec(self.AVAILABLE_OPS)
            if h not in count.keys():
                count[h] = 1
            else:
                count[h] += 1
        # print
        hashes, confidence = list(zip(*reversed(sorted(count.items(), key=itemgetter(1)))))
        return hashes, confidence

    def __str__(self):
        # s = 'Adjacent matrix: {} \n' \
        #     'Ops: {}\n'.format(self.matrix.tolist(), self.ops)
        s = 'alpha_topology: {} \n'.format(self.alpha_topology)
        s += 'alpha_ops: {} \n'.format(self.alpha_ops)
        return s


class NasBenchCellSearchDARTS(NasBenchCellSearch):
    # heavily support the weight sharing scheme in this layer.

    def __init__(self, input_channels, output_channels, model_spec, args=None):
        super(NasBenchCell, self).__init__() # nn.Module.init()
        assert isinstance(model_spec, ModelSpec_DARTS), "model spec to NASBenchCellSearch must be darts"
        self.model_spec = model_spec
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        # choose the vertex type.
        self.vertex_cls = MixedVertexDARTS
        # make a full connection to initialize this cell.
        self.change_model_spec(model_spec, initial=True)
        if self.has_skip:
            self.projection = conv_bn_relu(1, input_channels, output_channels)

    def change_model_spec(self, model_spec, initial=False, verbose=False):
        if not initial:
            raise ValueError("You cannot change DARTS Cell model spec. Only can be initialize once!")

        # Setup a graph structure to simplify our life
        dag = nx.from_numpy_matrix(model_spec.matrix, create_using=nx.DiGraph())
        node_labels = {}
        for i, op in enumerate(model_spec.ops):
            if op == "input" or op == "output":
                node_labels[i] = op
            else:
                node_labels[i] = "vertex_%d" % i
        dag = nx.relabel_nodes(dag, node_labels)
        # Resolve dependencies in graph
        self.execution_order = self._get_execution_order(dag)

        # Setup output_sizes for operations and assign vertex types
        out_shapes_list = compute_vertex_channels(
            self.input_channels, self.output_channels, model_spec.matrix
        )
        logging.debug('vertex channels %s', str(out_shapes_list))
        out_shapes = {}
        vertex_types = {}

        for t, (shape, op) in enumerate(zip(out_shapes_list, model_spec.ops)):
            out_shapes[node_labels[t]] = shape
            vertex_types[node_labels[t]] = op

        self.dag = dag
        # Setup the operations
        self.vertex_ops = nn.ModuleDict()
        for output_node, input_nodes in self.execution_order.items():
            if output_node == "output":
                continue
            # Setup all input shapes
            in_shapes = [out_shapes[node] for node in input_nodes]

            # Check if any of the inputs to the vertex comes form input to module
            is_input = [node == "input" for node in input_nodes]
            # Setup the operation
            # IPython.embed(header='')
            self.vertex_ops[output_node] = self.vertex_cls(
                in_shapes, out_shapes[output_node], vertex_types[output_node], is_input, args=self.args
            )

        # Handle skip connections to output
        self.has_skip = self.dag.has_edge("input", "output")
        if self.has_skip:
            # if len(self.execution_order['output']) > 1:
            self.execution_order["output"].remove("input")
            if len(self.execution_order['output']) == 0:
                del self.execution_order['output']

    def forward(self, inputs):
        x, ws = inputs
        # super(NasBenchCellSearchDARTS, self).forward()
        # Buffers to save intermediate resu
        tmp_buffers = {"input": x,
                       'output': 0.0}

        # Add output : 0.0 to avoid potential problem for [input, output] case.
        # Traverse based on predefined topological sort

        for output_node, input_nodes in self.execution_order.items():
            # Interior vertices are summed, outputs are concatenated
            if output_node != "output":
                in_buffers = [tmp_buffers[node] for node in input_nodes]
                node_id = eval(output_node.split('_')[1]) - 1
                weights = ws[node_id]
                tmp_buffers[output_node] = self.vertex_ops[output_node](in_buffers, weights)
            else:
                # We reverse the order to match tensorflow order for concatenation
                tmp_buffers[output_node] = torch.cat(
                    [tmp_buffers[ob] for ob in input_nodes], 1
                )

        # Becasue DARTS never zeros, so we always has skip.
        if self.has_skip:
            tmp_buffers["output"] += self.projection(x)

        return tmp_buffers["output"]


class NasBenchNetSearchDarts(NasBenchNet):
    AVAILABLE_OPS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

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
        logging.info("Creating DARTS model spec: {}".format(model_spec))

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

    def _loss(self, input, target):
        logits, _ = self(input, )
        return self._criterion(logits, target)

    def change_model_spec(self, model_spec, initial=False):
        assert isinstance(model_spec, ModelSpec_DARTS)
        # for cell in self.stacks.values():
        for k,v in self.stacks.items():
            # print("change spec for {}".format(k))
            for kk, vv in v.items():
                if 'module' in kk:
                    vv.change_model_spec(model_spec, initial)
        self._model_spec = model_spec
        # self._arch_parameters = nn.ParameterList(self.arch_parameters())
        # register parameters.
        return self

    def model_spec(self):
        return self._model_spec

    def summary(self):
        """
        Display the summary of a NasBenchSearch with respect to current model spec.
        :return:
        """
        # For stacks.
        for ind, stack in enumerate(self.stacks.values()):
            # because all stack module has the same architecture, only display the first module.
            logging.info(f'Stack {ind}: ')
            acell = stack['module0']
            acell.summary()

        logging.info("Current DARTS Spec {}".format(self._model_spec))

    def forward(self, x, model_spec=None, steps=None, bn_train=None):
        aux_logits = None
        # self._model_spec.update_multi_gpu(self.alpha_topology, self.alpha_ops)
        out = self.stem(x)
        ws = [self.weights(i) for i in range(self.num_intermediate_nodes)]
        for stack in self.stacks.values():
            for k, module in stack.items():
                # IPython.embed()
                if 'module' in k:
                    out = module([out, ws])
                else:
                    out = module(out)

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        # TO match the auxiliary head, but right now it is always false for NASBench.
        return self.dense(out), aux_logits

    def arch_parameters(self):
        return self.alpha_topology, self.alpha_ops

    def new(self, parallel=None):
        # new_model_spec = self._model_spec.new()
        model_new = NasBenchNetSearchDarts(**self.backup_kwargs)
        # model_new.change_model_spec(new_model_spec, initial=True)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.detach().clone())

        if self.args.gpus > 0:
            parallel_model = model_new.cuda()
        else:
            parallel_model = model_new
        return parallel_model


    def weights(self, node):
        """Sample weights for input node"""
        return self.topology_weights(node), self.ops_weights(node)

    def topology_weights(self, node):
        # return the soft weights for topology aggregation
        return nn.functional.softmax(self.alpha_topology[: node + 2, node], dim=0)[1:]

    def ops_weights(self, node):
        return nn.functional.softmax(self.alpha_ops[:, node], dim=0)

    def _sample_model_spec(self, num, sample_archs, sample_ops):
        new_model_specs = []
        with torch.no_grad():
            for _ in range(num):
                new_matrix = np.zeros((self.num_intermediate_nodes + 2, self.num_intermediate_nodes + 2), dtype=np.int)
                new_ops = ['input', ] + [None, ] * self.num_intermediate_nodes + ['output']
                for i in range(self.num_intermediate_nodes):
                    # action = 0 means, sample drop path
                    action = sample_archs[i].sample() - 1
                    if -1 < action < i + 1:  # cannot sample current node id
                        new_matrix[action, i + 1] = 1
                    # sample ops
                    op_action = sample_ops[i].sample()
                    new_ops[i + 1] = self.AVAILABLE_OPS[op_action]
                # logging.debug("Sampled architecture: matrix {}, ops {}".format(new_matrix, new_ops))
                new_matrix[:, -1] = 1  # connect all output
                new_matrix[-1, -1] = 0  # make it upper trianguler
                mspec = ModelSpec_v2(new_matrix, new_ops)
                # logging.debug('Sampled model spec {}'.format(mspec))
                new_model_specs.append(mspec)
                # mspec.hash_spec(self.AVAILABLE_OPS)
        return new_model_specs

    def sample_model_spec(self, num):
        """
        Sample model specs by number.
        :param num:
        :return: list, num x [architecture ]
        """
        alpha_topology = self.alpha_topology.detach().clone()
        alpha_ops = self.alpha_ops.detach().clone()
        sample_archs = []
        sample_ops = []
        with torch.no_grad():
            for i in range(self.num_intermediate_nodes):
                # align with topoligy weights
                probs = nn.functional.softmax(alpha_topology[: i+2, i], dim=0)
                sample_archs.append(Categorical(probs))
                probs_op = nn.functional.softmax(alpha_ops[:, i], dim=0)
                sample_ops.append(Categorical(probs_op))
            return self._sample_model_spec(num, sample_archs, sample_ops)

    def summary(self, num_sample):
        # This is to sample num_archs times, rank them by the number.
        count = {}
        model_specs = self.sample_model_spec(num_sample)

        for m in model_specs:
            h = m.hash_spec(self.AVAILABLE_OPS)
            if h not in count.keys():
                count[h] = 1
            else:
                count[h] += 1
        # print
        hashes, confidence = list(zip(*reversed(sorted(count.items(), key=itemgetter(1)))))
        return hashes, confidence

