from collections import defaultdict, deque



import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
# from torch.tensor import ones
from models.cnn_layers import CNN_LAYER_CREATION_FUNCTIONS, initialize_layers_weights, get_cnn_layer_with_names
from scipy.special import expit, logit
from typing import List

from models.shared_base import *
from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()


def node_to_key(node):
    idx, jdx, _type = node
    if isinstance(_type, str):
        return f'{idx}-{jdx}-{_type}'
    else:
        return f'{idx}-{jdx}-{_type.__name__}'


def dag_to_keys(dag):
    return [node_to_key(node) for node in dag]


class Architecture:
    """Represents some hyperparameters of the architecture requested.
    final_filter_size is the number of filters of the cell before the output layer.
    Each reduction filter doubles the number of filters (as it halves the width and height)
    There are num_modules modules stacked together.
    Each module except for the final one is made up of num_repeat_normal normal Cells followed by a reduction cell.
    The final layer doesn't have the reduction cell.
    """

    def __init__(self, final_filter_size, num_repeat_normal, num_modules):
        self.final_filter_size = final_filter_size
        self.num_repeat_normal = num_repeat_normal
        self.num_modules = num_modules


class CNN(SharedModel):
    """Represents a Meta-Convolutional network made up of Meta-Convolutional Cells.
       Paths through the cells can be selected and moved to the gpu for training and evaluation.

       Adapted from online code. need intense modification.

    """

    def __init__(self, args, corpus):
        """
                 # input_channels, height, width, output_classes, gpu, num_cell_blocks,
                 # architecture=Architecture(final_filter_size=768 // 2, num_repeat_normal=6, num_modules=3)):

        :param args: arguments
        :param corpus: dataset
        """
        super(CNN, self).__init__(args)
        self.args = args
        self.corpus = corpus

        architecture = Architecture(final_filter_size=args.cnn_final_filter_size,
                                    num_repeat_normal=args.cnn_num_repeat_normal,
                                    num_modules=args.cnn_num_modules)
        
        input_channels = args.cnn_input_channels
        self.height = args.cnn_height
        self.width = args.cnn_width
        self.output_classes = args.output_classes
        self.architecture = architecture

        self.output_height = self.height
        self.output_width = self.width
        self.num_cell_blocks = args.num_blocks

        self.cells = nn.Sequential()
        self.reduce_cells = nn.Sequential()
        self.normal_cells = nn.Sequential()

        self.gpu = torch.device("cuda:0") if args.num_gpu > 0 else torch.device('cpu')
        self.cpu_device = torch.device("cpu")

        self.dag_variables_dict = {}
        self.reducing_dag_variables_dict = {}

        last_input_info = _CNNCell.InputInfo(input_channels=input_channels, input_width=self.width)
        current_input_info = _CNNCell.InputInfo(input_channels=input_channels, input_width=self.width)

        # count connections
        temp_cell = _CNNCell(input_infos=[last_input_info, current_input_info],
                             output_channels=architecture.final_filter_size,
                             output_width=self.output_width, reducing=False, dag_vars=None,
                             num_cell_blocks=self.num_cell_blocks)

        self.all_connections = list(temp_cell.connections.keys()) # as all possible connections.

        self.dag_variables = torch.ones(len(self.all_connections), requires_grad=True, device=self.gpu)
        self.reducing_dag_variables = torch.ones(len(self.all_connections), requires_grad=True, device=self.gpu)

        for i, key in enumerate(self.all_connections):
            self.dag_variables_dict[key] = self.dag_variables[i]
            self.reducing_dag_variables_dict[key] = self.reducing_dag_variables[i]

        cells = [('normal', architecture.final_filter_size)] * architecture.num_repeat_normal
        current_filter_size = architecture.final_filter_size
        for module in range(architecture.num_modules - 1):
            cells.append(('reducing', current_filter_size))
            current_filter_size //= 2
            cells.extend([('normal', current_filter_size)] * architecture.num_repeat_normal)

        cells.reverse()

        for i, (type, num_filters) in enumerate(cells):
            if type == 'reducing':
                self.output_height /= 2
                self.output_width /= 2
                reducing = True
            else:
                reducing = False
                assert (type == 'normal')

            dag_vars = self.dag_variables_dict if reducing == False else self.reducing_dag_variables_dict
            name = f'{i}-{type}-{num_filters}'
            a_cell = _CNNCell(input_infos=[last_input_info, current_input_info],
                                           output_channels=num_filters, output_width=self.output_width,
                                           reducing=reducing, dag_vars=dag_vars, num_cell_blocks=self.num_cell_blocks,
                                           args=self.args)
            self.cells.add_module(name, a_cell)

            # Registering for the WPL later.
            if reducing:
                self.reduce_cells.add_module(name, a_cell)
            else:
                self.normal_cells.add_module(name, a_cell)

            last_input_info, current_input_info = current_input_info, _CNNCell.InputInfo(input_channels=num_filters,
                                                                                         input_width=self.output_width)

        if self.output_classes:
            self.conv_output_size = self.output_height * self.output_width * self.architecture.final_filter_size
            self.out_layer = nn.Linear(self.conv_output_size, self.output_classes)
            torch.nn.init.kaiming_normal_(self.out_layer.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(self.out_layer.bias, 0)
            self.out_layer.to(self.gpu)

        parent_counts = [0] * (2 + self.num_cell_blocks)

        for idx, jdx, _type in self.all_connections:
            parent_counts[jdx] += 1

        probs = np.array(list(2 / parent_counts[jdx] for idx, jdx, _type in self.all_connections))
        self.dags_logits = (logit(probs), logit(probs))

        self.target_ave_prob = np.mean(probs)
        self.cell_dags = ([], [])

        self.ignore_module_keys = ['cell', 'out_layer']
        self.wpl_monitored_modules = self.cells._modules
        self.init_wpl_weights()

    def forward(self, inputs,
                dag,
                is_train=True,
                hidden=None
                ):
        """
        :param cell_dags: (normal_cell_dag, reduction_cell_dag)
        :param inputs: [last_input, current_input]
        :param hidden: don't care. legacy for RNN.
        """

        cell_dag, reducing_cell_dag = dag or self.cell_dags
        # cell_dag, reducing_cell_dag = dag   # support the dynamic dags.

        is_train = is_train and self.args.mode in ['train'] # add here for behaviors differs from train and test.

        last_input, current_input = inputs, inputs

        for cell in self.cells:
            if cell.reducing:
                dag = reducing_cell_dag
            else:
                dag = cell_dag
            output, extra_out = cell(dag, last_input, current_input)
            last_input, current_input = current_input, output

        x = output.view(-1, self.conv_output_size)
        x = self.out_layer(x)
        return x, extra_out

    def get_f(self, name):
        """ Get the cell structure """
        name = name.lower()
        # return f
        raise NotImplementedError

    def get_num_cell_parameters(self, dag):
        """
        Returns the parameters of the path through the Meta-network given by the dag.
        :param dag: a list of [normal_dag, reduce_dag]
        return parameters.
        """
        dag, reducing_dag = dag
        params = []
        for cell in self.cells:
            if cell.reducing:
                d = reducing_dag
            else:
                d = dag
            params.extend(cell.get_parameters(d))
        # return params
        raise NotImplementedError

    def get_parameters(self, dags):
        """ return the parameter of given dags """
        dag, reducing_dag = dags
        params = []
        for cell in self.cells:
            if cell.reducing:
                d = reducing_dag
            else:
                d = dag
            params.extend(cell.get_parameters(d))
        return params

    def reset_parameters(self):
        """ reset all parameters ? """
        params = self.get_parameters(self.cell_dags)
        raise NotImplementedError('reset not implemented')

    def update_dag_logits(self, gradient_dicts, weight_decay, max_grad=0.1):
        """
        Updates the probabilities of each path being selected using the given gradients.
        """
        dag_probs = tuple(expit(logit) for logit in self.dags_logits)
        current_average_dag_probs = tuple(np.mean(prob) for prob in dag_probs)

        for i, key in enumerate(self.all_connections):
            for grad_dict, current_average_dag_prob, dag_logits in zip(gradient_dicts, current_average_dag_probs,
                                                                       self.dags_logits):
                if key in grad_dict:
                    grad = grad_dict[key] - weight_decay * (
                            current_average_dag_prob - self.target_ave_prob)  # *expit(dag_logits[i])
                    deriv = sigmoid_derivitive(dag_logits[i])
                    logit_grad = grad * deriv
                    dag_logits[i] += np.clip(logit_grad, -max_grad, max_grad)

    def get_dags_probs(self):
        """Returns the current probability of each path being selected.
        Each index corresponds to the connection in self.all_connections
        """
        return tuple(expit(logits) for logits in self.dags_logits)

    def __to_device(self, device, cell_dags):
        cell_dag, reducing_cell_dag = cell_dags
        for cell in self.cells:
            if cell.reducing:
                cell.to_device(device, reducing_cell_dag)
            else:
                cell.to_device(device, cell_dag)

    def set_dags(self, new_cell_dags=([], [])):
        """
        Sets the current active path. Moves other variables to the cpu to save gpu memory.

        :param new_cell_dags: (normal_cell_dag, reduction_cell_dag)
        """
        new_cell_dags = tuple(list(sorted(cell_dag)) for cell_dag in new_cell_dags)

        set_cell_dags = [set(cell_dag) for cell_dag in new_cell_dags]
        last_set_cell_dags = [set(cell_dag) for cell_dag in self.cell_dags]

        cell_dags_to_cpu = [last_set_cell_dag - set_cell_dag
                            for last_set_cell_dag, set_cell_dag in zip(last_set_cell_dags, set_cell_dags)]
        cell_dags_to_gpu = [set_cell_dag - last_set_cell_dag
                            for last_set_cell_dag, set_cell_dag in zip(last_set_cell_dags, set_cell_dags)]

        self.__to_device(self.cpu_device, cell_dags_to_cpu)
        self.__to_device(self.gpu, cell_dags_to_gpu)
        self.cell_dags = new_cell_dags

    # doing this is very important for grouping all the cells and unified the process.
    # maybe can move this to outer cells.
    # def init_wpl_weights(self):
    #     """
    #     Init for WPL operations.
    #
    #     NOTE: only take care of all the weights in self._modules, and others.
    #     for self parameters and operations, please override later.
    #
    #     :return:
    #     """
    #     for cell in self.cells:
    #         if isinstance(cell, WPLModule):
    #             cell.init_wpl_weights()
    #
    # def set_fisher_zero(self):
    #     for cell in self.cells:
    #         if isinstance(cell, WPLModule):
    #             cell.set_fisher_zero()
    #
    # def update_optimal_weights(self):
    #     """ Update the weights with optimal """
    #     for cell in self.cells:
    #         if isinstance(cell, WPLModule):
    #             cell.update_optimal_weights()

    def update_fisher(self, dags):
        """ logic is different here, for dags, update all the cells registered. """
        normal, reduce = dags
        for cell in self.cells:
            if cell.reducing:
                d = reduce
            else:
                d = normal
            cell.update_fisher(d)

    def compute_weight_plastic_loss_with_update_fisher(self, dags):
        loss = 0
        normal, reduce = dags
        for cell in self.cells:
            if cell.reducing:
                d = reduce
            else:
                d = normal
            loss += cell.compute_weight_plastic_loss_with_update_fisher(d)
        return loss


# Represents a Meta-Convolutional cell. It generates a possible forward connection between
# every layer except between the input layers of every type in CNN_LAYER_CREATION_FUNCTIONS
# Any path can then be chose to run and train with
class _CNNCell(WPLModule):

    class InputInfo:
        def __init__(self, input_channels, input_width):
            self.input_channels = input_channels
            self.input_width = input_width

    def __init__(self, input_infos: List[InputInfo],
                 output_channels, output_width,
                 reducing, dag_vars, num_cell_blocks,
                 args=None):
        super().__init__(args)

        self.input_infos = input_infos
        self.num_inputs = len(self.input_infos)
        self.num_cell_blocks = num_cell_blocks
        num_outputs = self.num_inputs + num_cell_blocks
        self.output_channels = output_channels
        self.output_width = output_width
        self.reducing = reducing
        self.dag_vars = dag_vars

        self.connections = dict()
        # self._connections = nn.ModuleList()

        for idx in range(num_outputs - 1):
            for jdx in range(max(idx + 1, self.num_inputs), num_outputs):
                for _type, type_name in get_cnn_layer_with_names():
                    if idx < self.num_inputs:
                        input_info = self.input_infos[idx]
                        if input_info.input_width != output_width:
                            assert (input_info.input_width / 2 == output_width)
                            stride = 2
                        else:
                            stride = 1
                        in_planes = input_info.input_channels

                    else:
                        stride = 1
                        in_planes = output_channels

                    out_planes = output_channels
                    try:
                        self.connections[(idx, jdx, type_name)] = _type(in_planes=in_planes, out_planes=out_planes,
                                                                        stride=stride)
                    except RuntimeError as e:
                        logger.error(f'Identity Matching error {e}')

                    initialize_layers_weights(self.connections[(idx, jdx, type_name)])
                    self.add_module(node_to_key((idx, jdx, type_name)), self.connections[(idx, jdx, type_name)])

        self.init_wpl_weights()

    def forward(self, dag, *inputs):
        """
        Define the actual CELL of one CNN structure.

        :param dag:
        :param inputs:
        :return:
            output: whatever output this mean
            extra_out: dict{string_keys}: to output additional variable/Tensors for regularization.
        """
        assert (len(inputs) == self.num_inputs)
        inputs = list(inputs)
        inputs = inputs + self.num_cell_blocks * [None]
        outputs = [0] * (self.num_inputs + self.num_cell_blocks)
        num_inputs = [0] * (self.num_inputs + self.num_cell_blocks)
        inputs_relu = [None] * (self.num_inputs + self.num_cell_blocks)

        for source, target, _type in dag:
            key = (source, target, _type)
            conn = self.connections[key]

            if inputs[source] is None:
                outputs[source] /= num_inputs[source]
                inputs[source] = outputs[source]
            layer_input = inputs[source]
            if hasattr(conn, 'input_relu') and conn.input_relu:
                if inputs_relu[source] is None:
                    inputs_relu[source] = torch.nn.functional.relu(layer_input)
                layer_input = inputs_relu[source]

            val = conn(layer_input) * self.dag_vars[key]
            outputs[target] += val
            num_inputs[target] += self.dag_vars[key]

        outputs[-1] /= num_inputs[-1]
        output = outputs[-1]
        raw_output = output

        extra_out = {'dropped': None,
                     'hiddens': None,
                     'raw': raw_output}

        return output, extra_out

    def to_device(self, device, dag):
        """Moves the parameters on the specified path to the device"""
        for source, target, type_name in dag:
            self.connections[(source, target, type_name)].to(device)

    def get_parameters(self, dag):
        """Returns the parameters of the path through the Cell given by the dag."""
        params = []
        for key in dag:
            params.extend(self.connections[key].parameters())
        return params

    def update_fisher(self, dag):
        """ a single dag"""
        super(_CNNCell, self).update_fisher(dag_to_keys(dag))

    def compute_weight_plastic_loss_with_update_fisher(self, dag):
        return super(_CNNCell, self).compute_weight_plastic_loss_with_update_fisher(dag_to_keys(dag))


def sigmoid_derivitive(x):
    """Returns the derivitive of a sigmoid function at x"""
    return expit(x) * (1.0 - expit(x))
