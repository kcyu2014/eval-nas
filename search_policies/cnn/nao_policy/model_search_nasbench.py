"""
Nasbench adaptation for NAO.
"""

import logging

import networkx as nx
import torch.nn as nn
import numpy as np

from ..search_space.nas_bench.model_builder import compute_vertex_channels
from nasbench.lib import config as _config
from search_policies.cnn.search_space.nas_bench.operations import conv_bn_relu, nasbench_vertex_weight_sharing
from ..search_space.nas_bench.model import NasBenchNet, NasBenchCell
import torch.nn.functional as F

NASBENCH_CONFIG = _config.build_config()


class NasBenchCellSearch(NasBenchCell):
    # heavily support the weight sharing scheme in this layer.

    def __init__(self, input_channels, output_channels, model_spec, args=None):
        super(NasBenchCell, self).__init__() # nn.Module.init()

        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        # choose the vertex type.
        self.vertex_cls = nasbench_vertex_weight_sharing[args.nasbenchnet_vertex_type]
        # make a full connection to initialize this cell.
        full_connection = 1 - np.tril(np.ones_like(model_spec.matrix))
        model_spec.matrix = full_connection
        self.change_model_spec(model_spec, initial=True)

        if self.has_skip:
            self.projection = conv_bn_relu(1, input_channels, output_channels)

    def summary(self):
        for ind, v in enumerate(self.vertex_ops.values()):
            # each vertex summarize itself.
            v.summary(ind)

    def change_model_spec(self, model_spec, initial=False, verbose=False):

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
        if verbose:
            logging.info('vertex channels %s', str(out_shapes_list))

        if initial:
            # generate the maximum possible channels.
            out_shapes_list = [self.input_channels,] + [self.output_channels,] * (len(out_shapes_list) - 1)

        out_shapes = {}
        vertex_types = {}

        for t, (shape, op) in enumerate(zip(out_shapes_list, model_spec.ops)):
            out_shapes[node_labels[t]] = shape
            vertex_types[node_labels[t]] = op

        self.dag = dag

        # print('node labels', node_labels)
        # print('out_shapes_list', out_shapes_list)
        # print('out_shapes', out_shapes)
        # print('vertex_types', vertex_types)
        # return node_labels, out_shapes, vertex_types, out_shapes_list

        # Setup the operations
        if initial:
            self.vertex_ops = nn.ModuleDict()
        for output_node, input_nodes in self.execution_order.items():
            if output_node == "output":
                continue
            # Setup all input shapes
            in_shapes = [out_shapes[node] for node in input_nodes]

            # Check if any of the inputs to the vertex comes form input to module
            is_input = [node == "input" for node in input_nodes]
            if initial:
                # Setup the operation
                self.vertex_ops[output_node] = self.vertex_cls(
                    in_shapes, out_shapes[output_node], vertex_types[output_node], is_input, self.args
                )
            else:
                # get the input_nodes order, by [input, vertex_i]
                input_nodes_id = [0 if x == 'input' else int(x.split('vertex_')[1]) for x in input_nodes]
                self.vertex_ops[output_node].change_vertex_type(
                    in_shapes, out_shapes[output_node], vertex_types[output_node],
                    input_nodes_id
                )

        # Handle skip connections to output
        self.has_skip = self.dag.has_edge("input", "output")
        if self.has_skip:
            # if len(self.execution_order['output']) > 1:
            self.execution_order["output"].remove("input")
            if len(self.execution_order['output']) == 0:
                del self.execution_order['output']
            # handle this case
                # del self.execution_order['output']


class NasBenchNetSearchNAO(NasBenchNet):

    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearch):
        """
        Weight sharing nasbench version 1.

        :param input_channels:
        :param model_spec:
        :param config: defined in nasbench.config
        """
        input_channels = 3
        model_spec = args.model_spec
        super(NasBenchNetSearchNAO, self).__init__(input_channels, model_spec, config, cell_cls, args)
        self.drop_path_keep_prob = self.args.child_drop_path_keep_prob
        self.dropout = nn.Dropout(1 - self.args.child_keep_prob)

    def change_model_spec(self, model_spec):
        # for cell in self.stacks.values():
        self._model_spec = model_spec
        for k,v in self.stacks.items():
            # print("change spec for {}".format(k))
            for kk, vv in v.items():
                if 'module' in kk:
                    vv.change_model_spec(model_spec)
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

    def forward(self, x, model_spec, steps=None, bn_train=None):
        aux_logits = None

        # we do not have two set of arch, only one.
        self.change_model_spec(model_spec)

        out = self.stem(x)
        for stack in self.stacks.values():
            for module in stack.values():
                out = module(out)

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        # TO match the auxiliary head, but right now it is always false for NASBench.
        return self.dense(out), aux_logits
