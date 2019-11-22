from collections import namedtuple

import torch
import ast

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
  'none',
  'max_pool_3x3',
  'avg_pool_3x3',
  'skip_connect',  # 3 identity
  'sep_conv_3x3',  # 4
  'sep_conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
  ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

TEST_NAONET_1276 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4)], normal_concat=[2, 3, 5, 6], reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4)], reduce_concat=[2, 3, 5, 6])

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
  normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
          ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
  normal_concat=[2, 3, 4, 5],
  reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
          ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
  reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2
NAO = TEST_NAONET_1276

def preprocess_arch(arch, config=None):
  if isinstance(arch, str):
    if config == 'enas':
      # cast list into archs[]
      l1, l2 = ast.literal_eval(arch)
      return l1 + l2
    else:
      arch = list(map(int, arch.strip().split()))
  elif isinstance(arch, list) and len(arch) == 2:
    arch = arch[0] + arch[1]
  return arch

def transfer_NAO_arch_to_genotype(arch):
  OP_MAPS = [PRIMITIVES[i] for i in [4, 5, 2, 1, 3]]
  arch = preprocess_arch(arch)
  normal_arch = arch[:int(len(arch) / 2)]
  reduce_arch = arch[int(len(arch) / 2):]
  nodes = int(len(normal_arch) / 4)
  normal_cell, normal_concat = process_arch_to_genotype(normal_arch, nodes, OP_MAPS)
  reduce_cell, reduce_concat = process_arch_to_genotype(reduce_arch, nodes, OP_MAPS)
  if 1 in reduce_concat:
    reduce_concat = [r for r in reduce_concat if r > 1]
  return Genotype(normal_cell, normal_concat, reduce_cell, reduce_concat)


def process_arch_to_genotype(arch, num_nodes, OP_MAPS):
  cell = []
  used = []
  for cell_id in range(num_nodes):
    x_id = arch[4 * cell_id]
    x_op = arch[4 * cell_id + 1]
    x_used = torch.zeros(num_nodes + 2).long()
    x_used[x_id] = 1
    cell.append((OP_MAPS[x_op], x_id))

    y_id = arch[4 * cell_id + 2]
    y_op = arch[4 * cell_id + 3]
    cell.append((OP_MAPS[y_op], y_id))
    y_used = torch.zeros(num_nodes + 2).long()
    y_used[y_id] = 1
    used.extend([x_used, y_used])

  used_ = torch.zeros(used[0].shape).long()
  for i in range(len(used)):
    used_ = used_ + used[i]
  indices = torch.eq(used_, 0).nonzero().long().view(-1)
  num_outs = indices.size(0)
  concat = indices.tolist()
  assert num_outs == len(concat)
  return cell, concat


def transfer_ENAS_arch_to_genotype(arch):
  OP_MAPS = [PRIMITIVES[i] for i in [4, 5, 2, 1, 3]]
  arch = preprocess_arch(arch, config='enas')
  arch_split_index = int(len(arch) / 2)
  normal_arch, reduce_arch = arch[:arch_split_index], arch[arch_split_index:]
  nodes = int(len(normal_arch)/ 4)
  # assert nodes == reduce_arch / 4
  normal_cell, normal_concat = process_arch_to_genotype(normal_arch, nodes, OP_MAPS)
  reduce_cell, reduce_concat = process_arch_to_genotype(reduce_arch, nodes, OP_MAPS)
  if 1 in reduce_concat:
    # adding to prevent problems
    reduce_concat = [r for r in reduce_concat if r > 1]
  return Genotype(normal_cell, normal_concat, reduce_cell, reduce_concat)


# Stores the results for 10 different runs.
NAO_FINAL_ARCHS = {1276: '0 1 0 1 1 1 0 1 0 1 1 1 1 1 4 1 1 1 4 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 4 1 1 1 4 1', 1277: '0 1 1 3 0 1 1 0 0 1 3 0 2 3 1 1 3 4 0 3 1 4 1 0 2 3 2 3 0 2 3 0 1 1 3 4 3 2 3 0', 1270: '0 4 0 1 0 1 0 1 0 3 0 4 2 4 1 0 1 0 1 1 0 1 1 1 1 0 1 3 2 2 3 3 1 0 0 3 4 3 4 3', 1273: '0 1 0 1 2 3 2 3 0 3 0 3 0 3 3 0 5 0 2 3 1 1 0 3 2 3 1 0 0 1 2 1 0 1 2 3 3 0 2 1', 1274: '0 4 0 1 0 1 0 4 0 4 0 1 4 0 3 3 4 2 4 3 1 4 1 3 0 4 0 3 2 3 0 1 0 1 4 0 5 3 3 2', 1268: '0 4 0 1 0 3 2 3 0 4 0 1 0 4 0 0 2 0 0 0 0 0 0 3 2 3 0 3 0 0 0 3 2 4 3 2 5 3 0 2', 1269: '0 1 0 0 0 4 1 0 1 2 1 3 2 2 0 3 2 4 0 4 1 0 0 4 0 1 2 2 2 0 1 3 4 2 1 3 5 2 3 0', 1272: '0 1 0 1 1 4 2 0 0 0 1 2 2 3 0 2 3 3 0 4 0 0 0 0 1 0 0 0 2 0 0 3 1 2 4 0 5 1 2 0', 1271: '0 1 0 0 0 1 0 3 3 1 3 4 0 0 0 0 4 0 3 2 0 2 0 2 0 2 2 1 2 2 2 4 0 1 2 3 2 2 4 0', 1275: '1 4 0 0 1 0 0 0 0 1 1 3 4 4 1 1 1 4 1 3 0 1 1 2 2 1 0 0 0 2 0 3 3 3 1 3 2 0 1 2'}
ENAS_FINAL_ARCHS = {1277: '[[0, 0, 0, 2, 0, 3, 1, 3, 0, 0, 0, 0, 2, 0, 1, 2, 3, 2, 0, 3], [0, 0, 1, 1, 2, 3, 1, 4, 2, 1, 1, 4, 3, 3, 0, 4, 5, 4, 3, 3]]', 1276: '[[0, 1, 1, 1, 1, 3, 0, 1, 0, 4, 0, 1, 0, 0, 1, 1, 0, 1, 1, 4], [1, 2, 1, 1, 1, 3, 0, 0, 1, 0, 1, 4, 0, 3, 1, 2, 1, 4, 0, 4]]', 1269: '[[0, 1, 0, 1, 0, 1, 0, 4, 0, 2, 2, 0, 0, 1, 0, 0, 0, 4, 0, 1], [0, 4, 1, 2, 0, 1, 1, 1, 0, 0, 3, 2, 2, 4, 2, 0, 4, 2, 2, 2]]', 1268: '[[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 0, 1, 3, 1, 0, 1, 0, 4], [1, 2, 0, 4, 0, 0, 1, 2, 1, 1, 2, 0, 4, 3, 4, 2, 0, 3, 4, 1]]', 1272: '[[1, 0, 0, 1, 0, 2, 0, 3, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0], [0, 1, 0, 1, 1, 4, 1, 1, 1, 1, 0, 1, 0, 2, 1, 2, 1, 2, 3, 4]]', 1273: '[[0, 4, 0, 4, 0, 2, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 4], [1, 2, 0, 1, 1, 3, 0, 2, 2, 3, 3, 4, 3, 2, 4, 4, 4, 3, 4, 0]]', 1271: '[[0, 1, 0, 1, 0, 4, 0, 0, 0, 1, 0, 1, 3, 2, 0, 0, 0, 1, 2, 3], [1, 0, 0, 1, 0, 1, 2, 2, 1, 0, 0, 1, 1, 4, 0, 1, 1, 2, 2, 1]]', 1270: '[[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2, 4, 1, 1, 1, 1, 1, 1], [0, 2, 1, 3, 1, 3, 0, 0, 0, 4, 3, 3, 1, 1, 1, 4, 0, 2, 1, 3]]', 1275: '[[0, 4, 0, 1, 0, 0, 0, 1, 0, 4, 2, 3, 0, 4, 0, 1, 0, 1, 0, 1], [0, 1, 1, 3, 1, 0, 2, 4, 0, 1, 0, 3, 0, 1, 3, 2, 0, 3, 5, 1]]', 1274: '[[0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 2, 0, 1, 0, 2, 4, 0, 0, 0], [1, 1, 0, 2, 0, 0, 2, 2, 2, 4, 2, 3, 1, 4, 0, 1, 4, 3, 2, 3]]'}
DARTS_FINAL_ARCHS = {1268: [[['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['avg_pool_3x3', 0], ['avg_pool_3x3', 1], ['skip_connect', 2], ['avg_pool_3x3', 0], ['skip_connect', 3], ['avg_pool_3x3', 0], ['avg_pool_3x3', 0], ['skip_connect', 3]], None],
                     1272: [[['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 2]], None, [['max_pool_3x3', 0], ['avg_pool_3x3', 1], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 2], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 3]], None],
                     1276: [[['sep_conv_3x3', 0], ['sep_conv_3x3', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['avg_pool_3x3', 1], ['skip_connect', 0], ['avg_pool_3x3', 1], ['skip_connect', 2], ['skip_connect', 2], ['max_pool_3x3', 1], ['skip_connect', 3], ['avg_pool_3x3', 1]], None],
                     1269: [[['sep_conv_3x3', 0], ['sep_conv_5x5', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 1]], None, [['avg_pool_3x3', 0], ['skip_connect', 1], ['avg_pool_3x3', 0], ['skip_connect', 2], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 2], ['avg_pool_3x3', 0]], None],
                     1273: [[['sep_conv_3x3', 0], ['sep_conv_3x3', 1], ['sep_conv_3x3', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['avg_pool_3x3', 1], ['avg_pool_3x3', 0], ['skip_connect', 2], ['avg_pool_3x3', 0], ['skip_connect', 3], ['avg_pool_3x3', 0], ['avg_pool_3x3', 0], ['skip_connect', 2]], None],
                     1271: [[['sep_conv_3x3', 0], ['sep_conv_3x3', 1], ['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['avg_pool_3x3', 0], ['avg_pool_3x3', 1], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 3], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 3]], None],
                     1277: [[['sep_conv_3x3', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['sep_conv_3x3', 1], ['skip_connect', 0], ['skip_connect', 1]], None, [['max_pool_3x3', 0], ['max_pool_3x3', 1], ['skip_connect', 2], ['max_pool_3x3', 0], ['max_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 2], ['avg_pool_3x3', 0]], None],
                     1270: [[['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['max_pool_3x3', 0], ['max_pool_3x3', 1], ['skip_connect', 2], ['max_pool_3x3', 0], ['avg_pool_3x3', 0], ['avg_pool_3x3', 1], ['skip_connect', 2], ['skip_connect', 3]], None],
                     1275: [[['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['sep_conv_3x3', 1], ['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 0], ['skip_connect', 1]], None, [['avg_pool_3x3', 0], ['skip_connect', 1], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 2], ['avg_pool_3x3', 0], ['avg_pool_3x3', 0], ['skip_connect', 3]], None],
                     1274: [[['sep_conv_3x3', 0], ['dil_conv_3x3', 1], ['skip_connect', 0], ['skip_connect', 2], ['sep_conv_3x3', 0], ['skip_connect', 2], ['skip_connect', 0], ['skip_connect', 2]], None, [['max_pool_3x3', 0], ['max_pool_3x3', 1], ['skip_connect', 2], ['avg_pool_3x3', 0], ['avg_pool_3x3', 0], ['skip_connect', 2], ['skip_connect', 3], ['skip_connect', 2]], None]}
RANDOM_V1_ARCHS = {1268: Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5]), 1269: Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('skip_connect', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5]), 1270: Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('dil_conv_3x3', 4), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5]), 1271: Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4), ('dil_conv_5x5', 4)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5]), 1272: Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5]), 1273: Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 3), ('avg_pool_3x3', 3), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5]), 1274: Genotype(normal=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5]), 1275: Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5]), 1276: Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('skip_connect', 4), ('dil_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5]), 1277: Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 3)], reduce_concat=[2, 3, 4, 5])}


def enas_final_genotypes_by_seed(seed):
  return transfer_ENAS_arch_to_genotype(ENAS_FINAL_ARCHS[seed])


def nao_final_genotypes_by_seed(seed):
  return transfer_NAO_arch_to_genotype(NAO_FINAL_ARCHS[seed])


def darts_final_genotype_by_seed(seed):
  arch = DARTS_FINAL_ARCHS[seed]
  return Genotype(arch[0], [2,3,4,5], arch[2], [2,3,4,5])


def random_generate_genotype_by_seed(seed):
  return RANDOM_V1_ARCHS[seed]
