import os
import math

import IPython
import numpy as np
import torch
import shutil
from torch.autograd import Variable

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]


class LRScheduler:
    def __init__(self, optimizer, args):
        self.last_lr_reset = 0
        self.lr_T_0 = args.child_lr_T_0
        self.child_lr_T_mul = args.child_lr_T_mul
        self.child_lr_min = args.child_lr_min
        self.child_lr_max = args.child_lr_max
        self.optimizer = optimizer

    def update(self, epoch):
        T_curr = epoch - self.last_lr_reset
        if T_curr == self.lr_T_0:
            self.last_lr_reset = epoch
            self.lr_T_0 = self.lr_T_0 * self.child_lr_T_mul
        rate = T_curr / self.lr_T_0 * math.pi
        lr = self.child_lr_min + 0.5 * (self.child_lr_max - self.child_lr_min) * (1.0 + math.cos(rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def generate_arch(n, num_nodes, num_ops=3):
    def _get_arch():
        arch = []
        for i in range(1, num_nodes + 1):
            p1 = np.random.randint(0, i)
            op1 = np.random.randint(0, num_ops)
            arch.extend([p1, op1])
        return arch

    archs = [_get_arch() for i in range(n)]  # [archs,]
    return archs


def build_dag(arch):
    if arch is None:
        return None
    # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
    arch = list(map(int, arch.strip().split()))
    # length = len(arch)
    return arch


def parse_arch_to_model_spec_matrix_op(cell, B=5):
    matrix = np.zeros((B + 2, B + 2))
    ops = [INPUT, ]
    is_output = [True, ] * (B + 1)
    try:
        for i in range(B):
            prev_node1 = cell[2 * i]  # O as input.
            op = ALLOWED_OPS[cell[2 * i + 1]]
            ops.append(op)

            is_output[prev_node1] = False
            curr_node = i + 1
            matrix[prev_node1][curr_node] = 1
        # process output
        for input_node, connect_to_output in enumerate(is_output):
            matrix[input_node][B + 1] = 1 if connect_to_output else 0
        matrix = matrix.astype(np.int)
        ops.append(OUTPUT)
    except Exception as e:
        IPython.embed()
    return matrix, ops


def parse_arch_to_seq(cell, branch_length, B=5):
    """
    Hard-coded mapping of architecture to this so called sequence.

    :param cell:
    :param branch_length:
    :param B:     So, B is number of intermeidate layer.
    :return:
    """
    assert branch_length in [2, ]
    seq = []

    # For Arch [], the only purpose is to encode [0-B] for the prev node and [B+1, ... B+ num_ops]
    for i in range(B):
        prev_node1 = cell[2 * i] + 1
        op1 = cell[2 * i + 1] + B + 1
        seq.extend([prev_node1, op1])
    return seq


def parse_seq_to_arch(seq, branch_length, B=5):
    n = len(seq)
    assert branch_length in [2, ]
    assert n // B == branch_length

    def _parse_cell(cell_seq):
        cell_arch = []
        for i in range(B):
            p1 = cell_seq[2 * i] - 1
            op1 = cell_seq[2 * i + 1] - (B + 2)
            cell_arch.extend([p1, op1])
        return cell_arch

    conv_seq = seq
    conv_arch = _parse_cell(conv_seq)
    return conv_arch


def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)

    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c

    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


def generate_eval_points(eval_epochs, stand_alone_epoch, total_epochs):
    if isinstance(eval_epochs, list):
        return eval_epochs
    assert isinstance(eval_epochs, int)
    res = []
    eval_point = eval_epochs - stand_alone_epoch
    while eval_point + stand_alone_epoch <= total_epochs:
        res.append(eval_point)
        eval_point += eval_epochs
    return res
