from copy import deepcopy

import IPython
import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import kendalltau
try:
    from nasbench.lib import base_ops
    from nasbench.lib import model_builder
except Exception as e:
    print(e)
    print("have trouble loading nasbench. ignore here.")
from functools import reduce
import logging
from .model import NasBenchNet
from .model_search import NasBenchNetSearch
import re
from search_policies.cnn.cnn_search_configs import build_default_args
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(nn.Module):
    """ Implement the linear regression for soft weightrs"""
    def __init__(self, models, target='weight'):
        super(LinearRegressionModel, self).__init__()
        self.models = models
        self.alphas = nn.ParameterDict()
        self.target = target
        self.model_params = {}
        for model in models:
            for name, p in model.named_parameters():
                if name not in self.model_params.keys():
                    self.model_params[name] = [p.data.detach().clone()]
                else:
                    self.model_params[name].append(p.data.detach().clone())
        for name, params in self.model_params.items():
            self.alphas.update({name.replace('.', '-'): nn.Parameter(torch.ones(len(params), dtype=torch.float32))})
        # copy all the params to non diff format and save in this model.

    def _apply(self, fn):
        super(LinearRegressionModel, self)._apply(fn)
        for k, params in self.model_params.items():
            self.model_params[k] = [fn(p) for p in params]
        return self

    def forward(self, x):
        if self.target == 'weight':
            return self.forward_on_weights(x)
        else:
            raise NotImplementedError("not now.")

    def forward_on_weights(self, x):
        """
        x is NamedParameters.
        :param x:
        :return:
        """
        loss = 0.
        loss_fn = nn.MSELoss()
        for name, p in x:
            stacked_p = torch.stack(self.model_params[name], dim=0)
            alpha_p = self.alphas[name.replace('.','-')].view([-1,] + [1,] * (stacked_p.dim() - 1))
            new_p = (alpha_p * stacked_p).sum(dim=0)
            loss += loss_fn(new_p, p)

        return loss

    def mse_solution(self, target_params):
        """ compute linear regression with fixed solution.d """
        loss = 0
        loss_fn = nn.MSELoss()
        for name, target in target_params:
            xs = stack_p = torch.stack(self.model_params[name], dim=0)
            xs = xs.view([xs.size(0), -1])
            t = target.view([1, -1])
            reg = LinearRegression(fit_intercept=False).fit(xs.detach().numpy().transpose(),
                                                            t.detach().numpy().transpose())
            self.alphas[name.replace('.', '-')].data.copy_(torch.tensor(reg.coef_[0], dtype=torch.float))

            alpha_p = self.alphas[name.replace('.', '-')].view([-1, ] + [1, ] * (stack_p.dim() - 1))
            new_p = (alpha_p * stack_p).sum(dim=0)
            loss += loss_fn(new_p, target).item()

        return loss

    def generate_parameters_with_alphas(self):
        new_params = {}
        for name, p_list in self.model_params.items():
            stacked_p = torch.stack(self.model_params[name], dim=0)
            alpha_p = self.alphas[name.replace('.', '-')].view([-1, ] + [1, ] * (stacked_p.dim() - 1))
            new_p = (alpha_p * stacked_p).sum(dim=0)
            new_params[name] = new_p
        return new_params


def load_nasbench_checkpoint_to_supernet(path, spec, legacy=True):
    """

    :param path: path to checkpoint
    :param spec: ModelSpec_v2
    :param legacy: Legacy mode, = true for old NasBenchNet, passed to load_nasbench_checkpoint.
    :return:
    """

    model, ckpt = load_nasbench_checkpoint(path, spec, legacy)
    default_args = build_default_args()
    default_args.model_spec = spec.new()
    model_search = NasBenchNetSearch(default_args)
    source_dict = model.state_dict()
    target_dict = model_search.state_dict()
    trans_dict= dict()

    for k in model.state_dict().keys():
        kk = transfer_model_key_to_search_key(k, spec)
        if kk not in model_search.state_dict().keys():
            print('not found ', kk)
            continue

        trans_dict[kk] = source_dict[k]
    # zero padding the dict.
    padded_dict = nasbench_zero_padding_load_state_dict_pre_hook(trans_dict, model_search.state_dict())
    model_search.load_state_dict(padded_dict)
    model_search = model_search.change_model_spec(spec)
    return model_search


def load_nasbench_checkpoint(path, spec, legacy=True):
    model = NasBenchNet(input_channels=3, model_spec=spec)
    ckpt = torch.load(path)
    state_dict = ckpt['model_state']
    # Transform the state_dict keys
    if legacy:
        trans_dict = dict()
        for k, v in state_dict.items():
            # map from the old version to newer.
            new_k = k.replace('.op.', '.PLC.')
            new_k = new_k.replace('.vertex_ops.', '.op.')
            if 'proj' in new_k:
                new_k = new_k.replace('.PLC.', '.vertex_ops.', 1)
                new_k = new_k.replace('.PLC.', '.op.', 1)
            else:
                new_k = new_k.replace('.PLC.', '.vertex_ops.', 1)
                new_k = new_k.replace('.PLC.', '.op.', 1)

            trans_dict[new_k] = v
    else:
        trans_dict = state_dict

    model.load_state_dict(trans_dict)
    return model, ckpt


def change_model_spec(model, spec):
    # this is handling parallel model
    """
    Change model spec, depends on Parallel model or not.
    :param model:
    :param spec:
    :return:
    """
    if isinstance(model, nn.DataParallel):
        model.module.change_model_spec(spec)
    else:
        model.change_model_spec(spec)
    return model

def display_cell(cell_data):
    for k, v in cell_data.items():
        logging.info('%s: %s' % (k, str(v)))


def deepgetattr(obj, attr):
    return reduce(getattr, attr, obj)


def tensorflow_param_to_pytorch(tf_val, oper_type, torch_module):
    def safe_param(tf_val, pt_val):
        if tf_val.shape != pt_val.shape:
            raise Exception(
                "TF to Pytorch copy: layer shapes don't match {} vs. {}".format(
                    tf_val.shape, pt_val.shape
                )
            )
        return nn.Parameter(torch.from_numpy(np.float64(tf_val)))

    if oper_type == "conv2d":
        torch_module[0].weight = safe_param(
            np.transpose(tf_val, [3, 2, 1, 0]), torch_module[0].weight
        )
    elif oper_type == "batch_normalization":
        torch_module[1].weight = nn.Parameter(torch.ones(torch_module[1].weight.shape))
        torch_module[1].bias = nn.Parameter(torch.zeros(torch_module[1].bias.shape))
    elif oper_type == "dense_weight":
        torch_module.weight = safe_param(tf_val.T, torch_module.weight)
    elif oper_type == "dense_bias":
        torch_module.bias = safe_param(tf_val, torch_module.bias)
    else:
        assert False


def nasbench_tensorflow_model_builder(model_spec, config, in_shape, is_training=True):
    if config["data_format"] == "channels_last":
        channel_axis = 3
    else:
        assert False

    # setup inputs
    features = tf.placeholder(tf.float32, shape=in_shape, name="g_input")

    # build the stem
    with tf.variable_scope("stem"):
        net = base_ops.conv_bn_relu(
            features, 3, config["stem_filter_size"], is_training, config["data_format"]
        )

    # Build stacks
    for stack_num in range(config["num_stacks"]):
        channels = net.get_shape()[channel_axis].value

        # Downsample at start (except first)
        if stack_num > 0:
            net = tf.layers.max_pooling2d(
                inputs=net,
                pool_size=(2, 2),
                strides=(2, 2),
                padding="same",
                data_format=config["data_format"],
            )

        # Double output channels each time we downsample
        channels *= 2

        with tf.variable_scope("stack{}".format(stack_num)):
            for module_num in range(config["num_modules_per_stack"]):
                with tf.variable_scope("module{}".format(module_num)):
                    net = model_builder.build_module(
                        model_spec,
                        inputs=net,
                        channels=channels,
                        is_training=is_training,
                    )

    # Global average pool
    if config["data_format"] == "channels_last":
        net = tf.reduce_mean(net, [1, 2])
    elif config["data_format"] == "channels_first":
        net = tf.reduce_mean(net, [2, 3])
    else:
        raise ValueError("invalid data_format")

    # Fully-connected layer to labels
    logits = tf.layers.dense(inputs=net, units=config["num_labels"])

    return features, logits


def copy_tensorflow_params_to_pytorch(pytorch_model, session):
    # Get trainable variables
    vars = tf.trainable_variables()
    vars_vals = session.run(vars)

    # Iterate over variables and find corresponding operations in Pytorch model
    for var, val in zip(vars, vars_vals):
        name_split = var.name.split("/")
        if name_split[0] == "stem":
            tensorflow_param_to_pytorch(val, name_split[1], pytorch_model.stem)
        elif name_split[0] == "dense":
            if name_split[1][:6] == "kernel":
                tensorflow_param_to_pytorch(val, "dense_weight", pytorch_model.dense)
            else:
                tensorflow_param_to_pytorch(val, "dense_bias", pytorch_model.dense)
        else:
            # Fetch pytorch model based on tf name
            stack = name_split[0]
            scope = name_split[1]
            vertex = name_split[2]
            vertex_type = name_split[3]
            operation = name_split[4]

            oper_ref = ["stacks", stack, scope]

            if vertex != "projection":
                oper_ref += ["op", vertex]

                if vertex_type == "projection":
                    oper_ref += ["proj_ops", "0"]
                else:
                    oper_ref += ["op"]
            else:
                oper_ref += [vertex]
                operation = vertex_type

            torch_module = deepgetattr(pytorch_model, oper_ref)

            tensorflow_param_to_pytorch(val, operation, torch_module)

    return pytorch_model.float()


##############################
# Projection
##############################


def transfer_model_key_to_search_key(k, model_spec):
    """
    transfer the keys from NasBenchNet to NasBenchNetSearch
    :param k:
    :return:
    """
    PROJ_OP_CHOICES = {'0': 'conv', '1': 'bn'}
    if 'proj_ops' in k:
        holder = k.split('.proj_ops.')[1][0:2]
        op_id = k.split('.proj_ops.')[1][2]
        # print(op_id)
        return k.replace(f'.proj_ops.{holder}{op_id}', f'.proj_ops.{holder}{PROJ_OP_CHOICES[op_id]}')
    elif 'vertex_ops.vertex' in k:
        # transform according to the model-spec
        # print('transfering ', k)

        vertex_id = int(k.split('.vertex_ops.vertex_')[1].split('.op.')[0])
        op_name = model_spec.ops[vertex_id]
        # print(vertex_id)
        # print('op is', op_name)
        prefix, suffix = k.split('.op.')
        suffix_op_id, suffix_rest = suffix.split('.')
        new_k = prefix + '.ops.' + op_name + '.' + PROJ_OP_CHOICES[suffix_op_id] + '.' + suffix_rest
        return new_k
    else:
        return k


def nasbench_zero_padding_load_state_dict_pre_hook(
    source_dict, target_dict, sanity_check=False
):
    target_dict = deepcopy(target_dict)
    for k, v in source_dict.items():
        if v.size() != target_dict[k].size():
            if '.bn.' in k:
                # 1d padding
                # print('diff before ', k,(target_dict[k][:v.size()[0]] - v).sum())
                target_dict[k][:v.size()[0]].data.copy_(v.detach().clone())
                # print('diff ', k ,(target_dict[k][:v.size()[0]] - v).sum())
            else:
                target_dict[k][:v.size()[0], :v.size()[1], :, :].data.copy_(v.detach().clone())
        else:
            target_dict[k].data.copy_(v.detach().clone())

    return target_dict


# sparse ranking things
def sparse_rank_by_simple_bin(ranks, perfs, threshold=1e-4, verbose=False):
    """
    Simple reduce the ranking by giving bin by threshold
    for example,
        threshold=1e=2, [0.911, 0.9112, 0.912] -> [1,1,2]
        threshold=1e=3, [0.911, 0.9112, 0.912] -> [1,2,3]

    :param ranks:
    :param perfs:
    :param threshold:
    :return:
    """
    new_ranks = []
    c_perf = 0.0
    c_rank = -1

    if threshold < 1.:
        multipler = 1 / threshold
    else:
        multipler = 1.

    for r, p in zip(ranks, perfs):
        p_rank = p * multipler // 1
        cp_rank = c_perf * multipler // 1
        if p_rank > cp_rank:
            c_rank += 1
        new_ranks.append(c_rank)
        c_perf = p
        if verbose:
            print(f'map ({p}) from {r} to {c_rank}')

    return new_ranks

def sort_hash_perfs(hashs, perfs, test=0, verbose=False):
    # make sure this is duplicated.
    import copy
    hashs = copy.deepcopy(hashs)
    perfs = copy.deepcopy(perfs)
    sorted_indices = np.argsort(perfs)
    if verbose:
        print(sorted_indices)
    s_h, s_p = [hashs[i] for i in sorted_indices], [perfs[i] for i in sorted_indices]
    return s_h, s_p


def rectify_model_ids_byvalid_to_bytest(model_ids, nasbench, hash_perfs):
    """
    Map from the model_id ranked by validation to ranked by testing.

    :param model_ids:
    :param nasbench:
    :param hash_perfs:
    :return:
    """
    # add logic if hash_perf is none
    if hash_perfs is None:
        raise NotImplementedError("Not yet supported. add this to nasbench later.")

    hashs = nasbench.hash_rank
    perfs = [hash_perfs[h] for h in hashs]
    hashs_bytest, perfs_bytest = sort_hash_perfs(hashs, [p[1] for p in perfs])
    hashs_byvalid, _ = sort_hash_perfs(hashs, [p[0] for p in perfs])
    rect_model_ids = [hashs_bytest.index(hashs_byvalid[i]) for i in model_ids]
    return rect_model_ids, [perfs_bytest[i] for i in rect_model_ids]


# function begins
def compute_sparse_kendalltau(model_ids, model_perfs, gt_perfs, threshold=1e-4,
                              fn_map_perf_to_new_rank=sparse_rank_by_simple_bin):
    """
    Compute the sparse kendall tau, by compression
    :param model_ids:
    :param model_perfs:
    :param gt_perfs:
    :param threshold:
    :param fn_map_perf_to_new_rank:
    :return:
    """
    avg_perfs_multiplier = 1e-2 if 1 < np.average(model_perfs) < 100 else 1.
    model_perfs = [p * avg_perfs_multiplier for p in model_perfs]
    gt_perfs, _ = sort_hash_perfs(gt_perfs, model_perfs)
    sgt_model_ids, sgt_perfs = sort_hash_perfs(model_ids, gt_perfs, verbose=False)
    sgt_sparse_ranks = fn_map_perf_to_new_rank(sgt_model_ids, sgt_perfs, threshold=threshold)
    pred_sparse_ranks = [sgt_sparse_ranks[sgt_model_ids.index(i)] for i in model_ids]
    print("Reduced ranks from {} to {}".format(len(set(model_ids)), len(set(sgt_sparse_ranks))))
    return kendalltau(sgt_sparse_ranks, pred_sparse_ranks)


def compute_percentile(model_ranks, num_total_arch, top_k=10, verbose=False):
    """
    Compute and print the percentile of a nasbench top models
    :param model_ranks: descending order, [0] is the best
    :param num_total_arch: Total number of archs to compute the percentile
    :param top_k: to reduce model ranks.
    :return:
    """
    r = np.asanyarray(model_ranks[:top_k], dtype=np.float)
    percentile = r / num_total_arch
    if verbose:
        print("Percentile of top {}: {} - {} - {}".format(top_k, percentile.min(), np.median(percentile), percentile.max()) )
        # should not compute the mean since percentile is more like categorical data.
    return percentile


class NasBenchZeroPaddingLoadStateDictPreHook(object):

    def __init__(self, original_shape):
        self.original_shape = original_shape

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass


class SoftWSNasBenchFit(LinearRegressionModel):

    def __init__(self):
        super(SoftWSNasBenchFit, self).__init__()
        pass

