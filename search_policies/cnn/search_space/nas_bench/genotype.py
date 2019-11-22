from nasbench import api

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

model_spec_demo = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
            [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


model_spec_full = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]],
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

model_spec_full_small = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1],    # input layer
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV3X3, CONV3X3, OUTPUT])

model_spec_linear = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]],
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV1X1, CONV1X1, CONV1X1, CONV1X1, OUTPUT])

model_spec_linear_small = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 0],    # input layer
            [0, 0, 1],
            [0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV3X3, OUTPUT])


model_spec_test = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0]],
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])


def test_model_max(spec):
    if spec == 1:
        return api.ModelSpec(
            # Adjacency matrix of the module
            matrix=[[0, 1, 0],    # input layer
                    [0, 0, 1],
                    [0, 0, 0]],   # output layer
            # Operations at the vertices of the module, matches order of matrix
            ops=[INPUT, MAXPOOL3X3, OUTPUT])
    elif spec == 2:
        return api.ModelSpec(
            # Adjacency matrix of the module
            matrix=[[0, 1, 1, 1],    # input layer
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]],   # output layer
            # Operations at the vertices of the module, matches order of matrix
            ops=[INPUT, MAXPOOL3X3, MAXPOOL3X3, OUTPUT])


