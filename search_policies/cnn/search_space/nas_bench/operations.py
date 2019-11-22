import itertools
import logging

import IPython
import torch
from torch.nn import SyncBatchNorm
from itertools import repeat

from functools import partial
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
from torch._six import container_abcs

from nasbench.lib import base_ops
from search_policies.cnn.nao_policy.operations import WSBN
from search_policies.cnn.operations import WSBNFull


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)

def conv_bn_relu(kernel_size, input_size, output_size):
    padding = 1 if kernel_size == 3 else 0

    out = nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(
            output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON
        ),
        nn.ReLU(inplace=False),
    )

    return out


def maxpool(kernel_size, *spargs):
    return nn.MaxPool2d(kernel_size, stride=1, padding=1)

OPS = {
        "conv1x1-bn-relu": partial(conv_bn_relu, 1),
        "conv3x3-bn-relu": partial(conv_bn_relu, 3),
        "maxpool3x3": partial(maxpool, 3),
    }


# Channel dropout functions.
def channel_select_dropout(x, out_size, dim):
    """ Channel dropout directly, rather than a layer. """
    orig_size = x.size()[dim]
    # move indices to GPU. this will be an interesting point.
    indices = torch.randperm(orig_size)[:out_size].to(x.device)
    return torch.index_select(x, dim, indices)


def channel_random_chunck(x, out_size, dim):
    orig_size = x.size()[dim]
    if orig_size - out_size == 0:
        return x
    start_channel = int(np.random.randint(0, orig_size - out_size))
    out = x.contiguous()
    return out.narrow(dim, start_channel, int(out_size))


def channel_chunck_fix(x, out_size, dim):
    out = x.contiguous()
    return out.narrow(dim, 0, int(out_size))


channel_drop_ops ={
    'fixed_chunck': channel_chunck_fix,
    'random_chunck': channel_random_chunck,
    'select_dropout': channel_select_dropout
}


class ChannelDropout(nn.Module):
    """
    Input:  [:, dim, :, :]
    Output: [:, out_size, :, :]
        out_size < dim, must be true.

    Various methods to test.

    """
    def __init__(self, channel_dropout_method='fixed', channel_dropout_dropouto=0.):
        super(ChannelDropout, self).__init__()
        self.channel_fn = channel_drop_ops[channel_dropout_method]
        self.dropout = nn.Dropout2d(channel_dropout_dropouto) if 0. < channel_dropout_dropouto < 1.0 else None

    def forward(self, input, output_size):
        # passed into a dropout layer and then get the
        # out = super(ChannelDropout, self).forward(input)
        if input.size()[1] - output_size <= 0:
            return input
        # out = input.contiguous()
        # start_channel = int(np.random.randint(0, input.size()[1] - output_size))
        # return out.narrow(1, start_channel, int(output_size))
        out = self.channel_fn(input, output_size, 1)
        if self.dropout:
            out = self.dropout(out)
        return out


class DynamicConv2d(nn.Conv2d):
    """
    Input: [:, dim, :, :], but dim can change.
    So it should dynamically adjust the self.weight.
    Using the similar logic as ChannelDropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 dynamic_conv_method='fixed', dynamic_conv_dropoutw=0.
                 ):
        self.channel_fn = channel_drop_ops[dynamic_conv_method]
        self.dropout = nn.Dropout2d(dynamic_conv_dropoutw) if 0. < dynamic_conv_dropoutw < 1.0 else None
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)

    def forward(self, input):
        batchsize, num_channels, height, width = input.size()
        # print("conv weight shape", self.weight.size())
        # print("conv channel size", self.weight.size()[1])
        # print("target channel ", num_channels)
        w = self.channel_fn(self.weight, num_channels, 1)
        if self.dropout:
            w = self.dropout(w)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DynamicReLUConvBN(nn.Module):
    """
    Conv Bn relu for weight sharing.
        Because the intrinsic of NasBench, for each cell,
            output_size = sum(intermediate_node.output_size)

        So to make weight sharing possible, we need to create a operation with full channel size then prune the channel.
        Each time, about the forward operation, we need to specify the output-size manually, for different architecture.

    """
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None):
        super(DynamicReLUConvBN, self).__init__()
        self.args = args
        padding = 1 if kernel_size == 3 else 0
        # assign layers.
        self.relu = nn.ReLU(inplace=False)
        self.conv = DynamicConv2d(
            full_input_size, full_output_size, kernel_size, padding=padding, bias=False,
            dynamic_conv_method=args.dynamic_conv_method, dynamic_conv_dropoutw=args.dynamic_conv_dropoutw
        )
        self.curr_vtx_id = curr_vtx_id
        tracking_stat = args.wsbn_track_stat
        if args.wsbn_sync:
            # logging.debug("Using sync bn.")
            self.bn = SyncBatchNorm(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                                    track_running_stats=tracking_stat)
        else:
            self.bn = nn.BatchNorm2d(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                                     track_running_stats=tracking_stat)

        self.bn_train = args.wsbn_train     # store the bn train of not.
        if self.bn_train:
            self.bn.train()
        else:
            self.bn.eval()

        # for dynamic channel
        self.channel_drop = ChannelDropout(args.channel_dropout_method, args.channel_dropout_dropouto)
        self.output_size = full_output_size
        self.current_outsize = full_output_size # may change according to different value.
        self.current_insize = full_input_size

    def forward(self, x, bn_train=False, output_size=None):
        # compute and only output the
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        # if self.bn_train and self.training:
        self.bn.train()
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_drop(x, output_size)
        return x

    def change_size(self, in_size, out_size):
        self.current_insize = in_size
        self.current_outsize = out_size


class DynamicConvWSBNRelu(DynamicReLUConvBN):

    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None):
        super(DynamicConvWSBNRelu, self).__init__(kernel_size, full_input_size, full_output_size, curr_vtx_id, args)
        del self.bn
        assert curr_vtx_id is not None and 0 <= curr_vtx_id < args.num_intermediate_nodes + 2, \
            logging.error("getting vertex id ", curr_vtx_id)
        self.bn = WSBNFull(curr_vtx_id + 1, full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        logging.debug('construction WSBN before creation, current id {}'.format(curr_vtx_id))
        self.previous_node_id = 0
        self.num_possible_inputs = curr_vtx_id + 1

    def change_previous_vertex_id(self, _id):
        """ change the id to use WSBN """
        if 0 <= _id < self.num_possible_inputs:
            self.previous_node_id = _id
        else:
            raise ValueError("Assigning previous id wrongly")

    def forward(self, x, bn_train=False, output_size=None):
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        # if self.bn_train and self.training:
        self.bn.train()
        x = self.bn(x, self.previous_node_id) # always using True for the moment.
        x = self.channel_drop(x, output_size)
        # x = self.relu(x)
        return x


class DynamicConvDifferentNorm(DynamicReLUConvBN):
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None, norm_type='instance'):
        super(DynamicConvDifferentNorm, self).__init__(kernel_size, full_input_size, full_output_size, curr_vtx_id, args)
        del self.bn
        if norm_type == 'instance':
            self.bn = nn.InstanceNorm2d(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        # elif norm_type == 'layer': # need to calculate the size, so abandon here.
        #     self.bn = nn.LayerNorm()
        elif norm_type == 'group':
            self.bn = nn.GroupNorm(num_groups=8, num_channels=full_output_size, eps=base_ops.BN_EPSILON)
        elif norm_type == 'local':
            self.bn = nn.LocalResponseNorm(2)
        else:
            raise ValueError("Norm type not yet supported ", norm_type)
        # self.bn = WSBNFull(curr_vtx_id + 1, full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        logging.debug('construction {} before creation, current id {}'.format(norm_type, curr_vtx_id))
        self.previous_node_id = 0
        self.num_possible_inputs = curr_vtx_id + 1


def maxpool_search(kernel_size, input_size, output_size, curr_vtx_id, args):
    return nn.MaxPool2d(kernel_size, stride=1, padding=1)


SEARCH_OPS = {
        "conv1x1-bn-relu": partial(DynamicReLUConvBN, 1),
        "conv3x3-bn-relu": partial(DynamicReLUConvBN, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class Truncate(nn.Module):
    def __init__(self, channels):
        super(Truncate, self).__init__()
        self.channels = channels

    def forward(self, x):
        return x[:, : self.channels]


class MixedVertex(nn.Module):

    oper = SEARCH_OPS

    def __init__(self, input_size, output_size, vertex_type, do_projection, args=None, curr_vtx_id=None):
        """

        :param input_size: input size to the cell, == in_channels for input node, and output_channels otherwise.
        :param output_size: full output size, should be the same as cell output_channels
        :param vertex_type: initial vertex type, choices over Ops.keys.
        :param do_projection: Projection with parametric operation. Set True for input and false for others
        :param args: args parsed into
        :param curr_vtx_id: current vertex id. used for advance logic for each cell.
        """
        super(MixedVertex, self).__init__()
        assert vertex_type in SEARCH_OPS.keys(), 'vertex must be inside OPS keys'
        self.args = args

        self.proj_ops = nn.ModuleList()
        # list to store the current projection operations, may change with model_spec
        self._current_proj_ops = []

        for ind, (in_size, do_proj) in enumerate(zip(input_size, do_projection)):
            proj_op = self.oper["conv1x1-bn-relu"](in_size, output_size, curr_vtx_id=curr_vtx_id, args=args) \
                if do_proj else Truncate(output_size)
            self.proj_ops.append(proj_op)
            self._current_proj_ops.append(ind)

        ops = nn.ModuleDict([])
        for k in SEARCH_OPS.keys():
            ops[k] = self.oper[k](output_size, output_size, curr_vtx_id=curr_vtx_id, args=args)

        self.ops = ops
        # self.op = self.oper[vertex_type](output_size, output_size)
        self.vertex_type = vertex_type
        self.output_size = output_size
        self.input_size = input_size

    def summary(self, vertex_id=None):
        """
        Summary this vertex, including nb channels and connection.
        Prefer output is:
        vertex {id}: [ id (channel), ... ] -> channel
        :return:
        """
        summary_str = f'vertex {vertex_id}:  ['
        for input_id in self._current_proj_ops:
            if input_id == 0:
                summary_str += f' 0 ({self.proj_ops[0].current_outsize})'
            else:
                summary_str += f' {input_id} ({self.proj_ops[input_id].channels})'
        summary_str += f'] - {self.vertex_type} -> {self.output_size}'
        logging.info(summary_str)
        return summary_str

    @property
    def current_op(self):
        return self.ops[self.vertex_type]

    @property
    def current_proj_ops(self):
        # dynamic compute the current project ops rather than store the pointer.
        return [self.proj_ops[p] for p in self._current_proj_ops]

    def change_vertex_type(self, input_size, output_size, vertex_type, proj_op_ids):
        # change vertex type accordingly.

        self._current_proj_ops = []
        for ind, (in_size, do_proj) in enumerate(zip(input_size, proj_op_ids)):
            if do_proj == 0: # Conv projection.
                self.proj_ops[do_proj].change_size(input_size, output_size)
            else:   # Truncate projection
                self.proj_ops[do_proj].channels = int(output_size)
                # print("Truncate output size ", output_size)

            # VERY IMPORTANT update the current proj ops list
            self._current_proj_ops.append(do_proj)

        if vertex_type in self.oper.keys():
            self.vertex_type = vertex_type
            # also update the current op
            # self.current_op = self.ops[vertex_type]
        else:
            raise ValueError("Update vertex_type error! Expected {} but got {}".format(
                self.oper.keys(), vertex_type
            ))

        for k, op in self.ops.items():
            # for convolutional layers, change the input/output size accordingly.
            if 'conv' in k:
                # logging.debug('vertex type', k, ' size to', output_size)
                op.change_size(output_size, output_size)
        self.output_size = output_size
        self.input_size = input_size

    def load_partial_parameters(self, ):
        pass

    def forward(self, x, weight=None):
        """
        Function it runs is
               relu(bn(w * \sum_i x_i + b))
        :param x:
        :param weight:
        :return:
        """
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
            # pass
        else:
            # This is the Single-path way.
            # apply all projection to get real input to this vertex
            proj_iter = iter(self.current_proj_ops)
            input_iter = iter(x)
            out = next(proj_iter)(next(input_iter))
            for proj, inp in zip(proj_iter, input_iter):
                out = out + proj(inp)
            return self.current_op(out)

    def trainable_parameters(self, prefix='', recurse=True):
        # print(f"compute trainable parameters, {self.vertex_type}, {self.current_proj_ops}")
        pf_d = {f'ops.{self.vertex_type}': self.current_op}
        for proj_id in self._current_proj_ops:
            pf_d[f'proj_ops.{proj_id}'] = self.proj_ops[proj_id]

        for pf, d in pf_d.items():
            for k, p in d.named_parameters(prefix=prefix + '.' + pf, recurse=recurse):
                yield k, p
#
#
# class MixedVertexNAO(MixedVertex):
#
#     def forward(self, x, weight=None):
#         if weight is not None and self.soft_aggregation:
#             raise NotImplementedError("To support DARTS way later.")
#             # pass
#         else:
#             # This is the Single-path way.
#             # apply all projection to get real input to this vertex
#             proj_iter = iter(self.current_proj_ops)
#             input_iter = iter(x)
#             out = next(proj_iter)(next(input_iter))
#             for proj, inp in zip(proj_iter, input_iter):
#                 out = out + proj(inp)
#             output = self.current_op(out)
#             # if self.vertex_type != "maxpool3x3":
#             #     output = self.apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
#             return output
#
#     @staticmethod
#     def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
#         layer_ratio = float(layer_id+1) / (layers)
#         drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
#         step_ratio = float(step + 1) / steps
#         drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
#         if drop_path_keep_prob < 1.:
#             mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
#             x.div_(drop_path_keep_prob)
#             x.mul_(mask)
#             x = x / drop_path_keep_prob * mask
#         return x


class MixedVertexDARTS(MixedVertex):

    def forward(self, x, weight):
        """

        :param x: input.
        :param weight: topology, op weights accordingly
        :return:
        """
        topo_weight, op_weight = weight
        # This is the Single-path way.
        # apply all projection to get real input to this vertex
        proj_iter = iter(self.current_proj_ops)
        input_iter = iter(x)
        out = next(proj_iter)(next(input_iter))
        for ind, (proj, inp) in enumerate(zip(proj_iter, input_iter)):
            out = out + topo_weight[ind] * proj(inp)
        # no such thing called current op.
        output = 0.0
        try:
            for ind, op in enumerate(self.ops.values()):
                output = output + op_weight[ind] * op(out)
        except RuntimeError as e:
            IPython.embed()
        return output


class MixedVertexWSBN(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvWSBNRelu, 1),
        "conv3x3-bn-relu": partial(DynamicConvWSBNRelu, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

    def forward(self, x, weight=None):
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
        else:
            # version 1. this is like
            #       relu( \sum_i{bn_i ( w x_i + b_i))} )
            out = 0.
            for in_id, inp in zip(self._current_proj_ops, x):
                if 'conv' in self.vertex_type:
                    self.current_op.change_previous_vertex_id(in_id)
                out = out + self.current_op(self.proj_ops[in_id](inp))
            return F.relu(out)


class MixedVertexWSBNSumEnd(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvWSBNRelu, 1),
        "conv3x3-bn-relu": partial(DynamicConvWSBNRelu, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

    def forward(self, x, weight=None):
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
        else:
            # version 2. this is like
            #        \sum_i relu({bn_i ( w x_i + b_i))})
            out = 0.
            for in_id, inp in zip(self._current_proj_ops, x):
                if 'conv' in self.vertex_type:
                    self.current_op.change_previous_vertex_id(in_id)
                out = out + F.relu(self.current_op(self.proj_ops[in_id](inp)))
            return out


class MixedVertexInstanceNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='instance'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='instance'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class MixedVertexGroupNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='group'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='group'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class MixedVertexLRNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='local'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='local'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


nasbench_vertex_weight_sharing = {
    'mixedvertex':     MixedVertex,
    'mixedvertex_instance_norm':     MixedVertexInstanceNorm,
    'mixedvertex_group_norm':     MixedVertexGroupNorm,
    'mixedvertex_local_norm':     MixedVertexLRNorm,
    'mixedvertex_wsbn':     MixedVertexWSBN,
    'mixedvertex_wsbn_endsum':     MixedVertexWSBNSumEnd,
    'nao_ws':  MixedVertex,
    'enas_ws': MixedVertex,
    'darts_ws': MixedVertexDARTS,
    'fbnet_ws': MixedVertexDARTS,
    'spos_ws': MixedVertex,
    'fairnas_ws': MixedVertex,
}
