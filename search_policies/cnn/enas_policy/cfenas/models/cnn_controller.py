"""
Implement the CNN controller because of the following differences.

CNN RNN-sampler structure is

    For micro-search, each node, sample 2 previous node, and two operations.

RNN RNN-sampler produces the
    len(prev_nodes) = num_blocks
    len(activations) = num_blocks

while micro-search CNN is

    len(prev_nodes) = 2 * num_blocks
    len(activations) = 2 * num_blocks

Pytorch CNN implementation requires the format


DAG structure:

For MicroCNN Cell:

    dags = [
        [id1, id2, op_name], ...
    ]

    NOTE: id=0 represent for the layer(i-1), id=1 represents input to current layer.



"""
import collections
import os

import torch
import torch.nn.functional as F
import utils

from .controller import Controller


# MicroNode = collections.namedtuple('MicroNode', ['id1, id2', 'op1', 'op2'])
MicroNode = collections.namedtuple('MicroNode', ['id1', 'id2', 'op1'])
MicroArchi = collections.namedtuple('MicroArchi', ['normal_cell', 'reduced_cell'])


class CNNMicroController(Controller):
    """ define a single controller for one type of cell. """

    def __init__(self, args):
        super(CNNMicroController, self).__init__(args)
        self.args = args

        if self.args.network_type == 'micro_cnn':
            # First node always take the input to the Cell.

            # For normal cell
            # self.num_normal_tokens = [len(args.shared_cnn_normal_types)]
            self.num_normal_tokens = []
            for idx in range(self.args.num_blocks):
                # NOTE for CNN, the node have two input and op rather than 1.
                self.num_normal_tokens += [idx + 2, len(args.shared_cnn_normal_types)] * 2
            self.normal_func_names = args.shared_cnn_normal_types

            # reduce_tokens
            # self.num_reduce_tokens = [len(args.shared_cnn_reduce_types)]
            self.num_reduce_tokens = []
            for idx in range(self.args.num_blocks):
                self.num_reduce_tokens += [idx + 2, len(args.shared_cnn_reduce_types)] * 2
            self.reduce_func_names = args.shared_cnn_reduce_types

            # Combine the num tokens as a full list.
            self.num_tokens = self.num_normal_tokens + self.num_reduce_tokens
            self.func_names = [self.normal_func_names, self.reduce_func_names]
        else:
            raise NotImplementedError(f'{self.args.network_type} is not supported yet')

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid)

        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def sample(self, batch_size=1, with_details=False, save_dir=None, construct_dag_method=None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        def _construct_micro_cnn_dags(prev_nodes, activations, func_names, num_blocks):
            """Constructs a set of DAGs based on the actions, i.e., previous nodes and
            activation functions, sampled from the controller/policy pi.

            This will be tailored for CNN only. Not the afore-mentioned RNN.

            Args:
                prev_nodes: Previous node actions from the policy.
                activations: Activations sampled from the policy.
                func_names: [normal_func_names, reduce_func_names]
                num_blocks: Number of blocks in the target RNN cell.

            Returns:
                A list of DAGs defined by the inputs.

            CNN cell DAGs are represented in the following way:

            1. entire DAG is represent as a simple list, of element 2
                [ Normal-Cell, Reduction-Cell ]
            2. each element is another list, containing such information
                [ (node_id1, node_id2, ops), ] * num_blocks
                    represents node1 -- ops --> node 2

            3. node 0, represents the h(t-1), i.e. previous layer input
               node 1, represents the h(t), i.e. current input
                    so, the actually index for current block starts from2

            """
            dags = []
            for nodes, func_ids in zip(prev_nodes, activations):
                dag = []

                # compute the first node
                # dag.append(MicroNode(0, 2, func_names[func_ids[0]]))
                # dag.append(MicroNode(1, 2, func_names[func_ids[0]]))
                leaf_nodes = set(range(2, num_blocks + 2))

                # add following nodes
                for curr_idx, (prev_idx, func_id) in enumerate(zip(nodes, func_ids)):
                    layer_id = curr_idx // 2 + 2
                    _prev_idx = utils.to_item(prev_idx)
                    if _prev_idx == layer_id:
                        continue
                    assert _prev_idx < layer_id, "Crutial logical error"
                    dag.append(MicroNode(_prev_idx, layer_id, func_names[func_id]))
                    leaf_nodes -= set([_prev_idx])

                # add leaf node connection with concat
                # for idx in leaf_nodes:
                #     dag.append(MicroNode(idx, num_blocks, 'concat'))
                dag.sort()
                dags.append(dag)

            return dags

        construct_dag_method = construct_dag_method or _construct_micro_cnn_dags

        list_dags = []
        final_log_probs = []
        final_entropies = []
        block_num = 4 * self.args.num_blocks
        # Iterate Normal cell and Reduced cell
        for type_id in range(2):

            if batch_size < 1:
                raise Exception(f'Wrong batch_size: {batch_size} < 1')

            # [B, L, H]
            inputs = self.static_inputs[batch_size]
            hidden = self.static_init_hidden[batch_size]

            activations = []
            entropies = []
            log_probs = []
            prev_nodes = []

            for block_idx in range((0 + type_id) * block_num, (1 + type_id) * block_num):
                logits, hidden = self.forward(inputs,
                                              hidden,
                                              block_idx,
                                              is_embed=(block_idx == (0 + type_id) * block_num))

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(log_prob * probs).sum(1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(
                    1, utils.get_variable(action, requires_grad=False))

                # .view()? Same below with `action`.
                entropies.append(entropy)
                log_probs.append(selected_log_prob[:, 0])

                # 1: function, 0: previous node
                mode = block_idx % 2
                inputs = utils.get_variable(
                    action[:, 0] + sum(self.num_tokens[:mode]),
                    requires_grad=False)

                if mode == 1:
                    activations.append(action[:, 0])
                elif mode == 0:
                    prev_nodes.append(action[:, 0])

            prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
            activations = torch.stack(activations).transpose(0, 1)

            dags = construct_dag_method(prev_nodes,
                                        activations,
                                        self.normal_func_names if type_id == 0 else self.reduce_func_names,
                                        self.args.num_blocks)
            if save_dir is not None:
                for idx, dag in enumerate(dags):
                    utils.draw_network(dag,
                                       os.path.join(save_dir, f'graph{idx}.png'))
            # add to the final result
            list_dags.append(dags)
            final_entropies.extend(entropies)
            final_log_probs.extend(log_probs)

        list_dags = [MicroArchi(d1, d2) for d1, d2 in zip(list_dags[0], list_dags[1])]

        if with_details:
            return list_dags, torch.cat(final_log_probs), torch.cat(final_entropies)

        if batch_size == 1 and len(list_dags) != 1:
            list_dags = [list_dags]
        elif batch_size != len(list_dags):
            raise RuntimeError(f"Sample batch_size {batch_size} does not match with len list_dags {len(list_dags)}")
        return list_dags
