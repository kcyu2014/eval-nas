import IPython
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from nasbench.lib import config as _config
from search_policies.cnn.search_space.nas_bench.model_builder import compute_vertex_channels
from .operations import conv_bn_relu, OPS
NASBENCH_CONFIG = _config.build_config()


class Truncate(nn.Module):
    def __init__(self, channels):
        super(Truncate, self).__init__()
        self.channels = channels

    def forward(self, x):
        return x[:, : self.channels]


class Vertex(nn.Module):
    """
    The weights of operations lies in this vertex node.
    """
    oper = OPS

    def __init__(self, input_size, output_size, vertex_type, do_projection):
        """

        :param input_size:
        :param output_size:
        :param vertex_type: type is the output node.
        :param do_projection:
        """
        super(Vertex, self).__init__()

        self.proj_ops = nn.ModuleList()

        for in_size, do_proj in zip(input_size, do_projection):
            if do_proj:
                self.proj_ops.append(self.oper["conv1x1-bn-relu"](in_size, output_size))
            else:
                self.proj_ops.append(Truncate(output_size))

        self.op = self.oper[vertex_type](output_size, output_size)

    def forward(self, x):
        proj_iter = iter(self.proj_ops)
        input_iter = iter(x)

        out = next(proj_iter)(next(input_iter))
        for proj, inp in zip(proj_iter, input_iter):
            out = out + proj(inp)

        return self.op(out)


class NasBenchCell(nn.Module):

    def __init__(self, input_channels, output_channels, model_spec, args=None):
        super(NasBenchCell, self).__init__()
        self.args = args

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
            input_channels, output_channels, model_spec.matrix
        )
        # print('vertex channels :', out_shapes_list)
        out_shapes = {}
        vertex_types = {}
        for t, (shape, op) in enumerate(zip(out_shapes_list, model_spec.ops)):
            out_shapes[node_labels[t]] = shape
            vertex_types[node_labels[t]] = op

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
            self.vertex_ops[output_node] = Vertex(
                in_shapes, out_shapes[output_node], vertex_types[output_node], is_input
            )

        # Handle skip connections to output
        self.has_skip = dag.has_edge("input", "output")
        if self.has_skip:
            # if len(self.execution_order['output']) > 1:
                # special case for [input, output]
            self.execution_order["output"].remove("input")
            self.projection = conv_bn_relu(1, input_channels, output_channels)
            if len(self.execution_order['output']) == 0:
                del self.execution_order['output']

        self.dag = dag

    def forward(self, x):
        # Buffers to save intermediate results
        tmp_buffers = {"input": x,
                       'output': 0.0}
        # Add output : 0.0 to avoid potential problem for [input, output] case.

        # Traverse based on predefined topological sort
        for output_node, input_nodes in self.execution_order.items():
            # Interior vertices are summed, outputs are concatenated
            if output_node != "output":
                in_buffers = [tmp_buffers[node] for node in input_nodes]
                tmp_buffers[output_node] = self.vertex_ops[output_node](in_buffers)
            else:
                # We reverse the order to match tensorflow order for concatenation
                tmp_buffers[output_node] = torch.cat(
                    [tmp_buffers[ob] for ob in input_nodes], 1
                )
        # At input-output skip conection if necessary
        if self.has_skip:
            tmp_buffers["output"] += self.projection(x)

        return tmp_buffers["output"]

    def forward_debug(self, x):
        # Buffers to save intermediate results
        tmp_buffers = {"input": x,
                       'output': 0.0}
        # Add output : 0.0 to avoid potential problem for [input, output] case.

        # Traverse based on predefined topological sort
        for output_node, input_nodes in self.execution_order.items():
            # Interior vertices are summed, outputs are concatenated
            if output_node != "output":
                in_buffers = [tmp_buffers[node] for node in input_nodes]
                tmp_buffers[output_node] = self.vertex_ops[output_node](in_buffers)
            else:
                # We reverse the order to match tensorflow order for concatenation
                tmp_buffers[output_node] = torch.cat(
                    [tmp_buffers[ob] for ob in input_nodes], 1
                )
        # At input-output skip conection if necessary
        if self.has_skip:
            tmp_buffers["output"] += self.projection(x)

        return tmp_buffers

    # Build execution order
    def _get_execution_order(self, dag):
        order = OrderedDict()
        for vert in nx.topological_sort(dag):
            _input = list(dag.predecessors(vert))
            if len(_input) > 0:
                # only build order with more than 0 input node.
                order[vert] = _input
        # del order["input"]
        # print('order dict', order.keys())
        return order


class NasBenchNet(nn.Module):

    def __init__(self, input_channels, model_spec, config=NASBENCH_CONFIG, cell_cls=NasBenchCell, args=None):
        """
        Declaration of a NASBenchNet.
        Checked with TF Code by Rene.

        It is very important that not changing the core logic, e.g., num stacks or other.

        :param input_channels:
        :param model_spec:
        :param config: defined in nasbench.config, can be overwritten.

        """

        super(NasBenchNet, self).__init__()
        self.args = args     # global configs
        stem_channels = config["stem_filter_size"]
        self.stem = conv_bn_relu(3, input_channels, stem_channels)
        stacks = nn.ModuleDict([])

        input_channels = stem_channels
        for stack_num in range(config["num_stacks"]):
            output_channels = 2 * input_channels

            stack = nn.ModuleDict(
                [("downsample", self._get_downsampling_operator())]
                if stack_num > 0
                else []
            )

            for module_num in range(config["num_modules_per_stack"]):
                module_name = "module%d" % module_num
                module = cell_cls(input_channels, output_channels, model_spec, args)
                stack.update([(module_name, module)])
                input_channels = output_channels

            stack_name = "stack%d" % stack_num
            stacks.update([(stack_name, stack)])

        self.stacks = stacks
        self.dense = nn.Linear(output_channels, 10)
        self._model_spec = model_spec

    def forward_debug(self, x):
        temp_outs = []
        out = self.stem(x)
        temp_outs.append(out.detach().clone().cpu())
        for stack in self.stacks.values():
            for module in stack.values():
                out = module(out)
                temp_outs.append(out.detach().clone().cpu())

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        # TO match the auxiliary head.
        temp_outs.append(out.detach().clone().cpu())
        return temp_outs

    def forward(self, x):
        # temp_outs = []
        out = self.stem(x)
        # temp_outs.append(out.detach().clone())
        for stack in self.stacks.values():
            for module in stack.values():
                out = module(out)
                # temp_outs.append(out.detach().clone())

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        # TO match the auxiliary head.
        # temp_outs.append(out)
        # return temp_outs
        return self.dense(out), None

    def _get_downsampling_operator(self):
        operator = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(2, stride=2, padding=0)
        )

        return operator
