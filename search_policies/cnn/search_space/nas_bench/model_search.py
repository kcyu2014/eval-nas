import logging

import IPython
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from search_policies.cnn.search_space.nas_bench.sampler import obtain_full_model_spec
from .model_builder import compute_vertex_channels
from nasbench.lib import config as _config
from .operations import conv_bn_relu, nasbench_vertex_weight_sharing
from .model import NasBenchNet, NasBenchCell


NASBENCH_CONFIG = _config.build_config()



class NasBenchCellSearch(NasBenchCell):
    # heavily support the weight sharing scheme in this layer.

    def __init__(self, input_channels, output_channels, model_spec=None, args=None):
        super(NasBenchCell, self).__init__() # nn.Module.init()

        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        # choose the vertex type.
        self.vertex_cls = nasbench_vertex_weight_sharing[args.nasbenchnet_vertex_type]
        # make a full connection to initialize this cell.
        model_spec = model_spec or obtain_full_model_spec(args.num_intermediate_nodes + 2)
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
            logging.info('vertex channels %s', str(out_shapes_list))

        out_shapes = {}
        vertex_types = {}

        for t, (shape, op) in enumerate(zip(out_shapes_list, model_spec.ops)):
            out_shapes[node_labels[t]] = shape
            vertex_types[node_labels[t]] = op

        self.dag = dag

        # if not initial:
        #     IPython.embed()

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
            # if len(input_nodes) == 0:
            #     continue

            # Setup all input shapes
            in_shapes = [out_shapes[node] for node in input_nodes]

            # Check if any of the inputs to the vertex comes form input to module
            is_input = [node == "input" for node in input_nodes]
            if initial:
                # Setup the operation
                output_node_id = int(output_node.split('vertex_')[1])
                self.vertex_ops[output_node] = self.vertex_cls(
                    in_shapes, out_shapes[output_node], vertex_types[output_node], is_input,
                    curr_vtx_id=output_node_id, args=self.args
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

    def trainable_parameters(self, prefix='', recurse=True):
        for k, m in self.vertex_ops.items():
            if isinstance(m, self.vertex_cls):
                # print(k)
                # print(m)
                for k, p in m.trainable_parameters(f'{prefix}.vertex_ops.{k}', recurse):
                    yield k, p
        if self.has_skip:
            for k, p in self.projection.named_parameters(f'{prefix}.projection', recurse):
                yield k, p


class NasBenchNetSearch(NasBenchNet):

    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearch):
        """
        Weight sharing nasbench version 1.

        :param input_channels:
        :param model_spec:
        :param config: defined in nasbench.config
        """
        input_channels = 3
        model_spec = args.model_spec
        super(NasBenchNetSearch, self).__init__(input_channels, model_spec, config, cell_cls, args)

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

    def trainable_parameters(self):
        """
        :return: trainable parameters that will be used in stack.forward
        """
        for k, v in self.stacks.items():
            for kk, vv in v.items():
                prefix = f'stacks.{k}.{kk}'
                if hasattr(vv, 'trainable_parameters'):
                    yield vv.trainable_parameters(prefix=prefix)
                else:
                    yield vv.named_parameters(prefix)
                    # yield vv.named_parameters()


class NasBenchCellSearchSoftWS(NasBenchCellSearch):

    def trainable_parameters(self):
        """
        Trainable parameters here include soft-weight-alpha and the parameters.
        :return:
        """
        pass

class NasBenchNetSearchSoftWeightSharing(NasBenchNetSearch):
    # this is to implement the soft-weight-sharing cell.
    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearchSoftWS):
        super(NasBenchNetSearchSoftWeightSharing, self).__init__()
