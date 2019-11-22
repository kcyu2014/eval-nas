"""
Goal is to sample the architecture according to its rank.

"""

# The first step is to collect all the possible architecture. i should make a notebook for this?
# Standard imports
import copy
import json

import IPython
import numpy as np
import matplotlib.pyplot as plt
import random
from nasbench import api


# Useful constants
from .nasbench_api_v2 import ModelSpec_v2
from .genotype import CONV3X3, INPUT, OUTPUT

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
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


def random_spec(nasbench):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return spec


# def random_to_nas_to_random(nasbench):
#     model_spec = nasb


def mutate_spec(nasbench, old_spec, mutation_rate=1.0):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / NUM_VERTICES
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                if random.random() < edge_mutation_prob:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / OP_SPOTS
        for ind in range(1, NUM_VERTICES - 1):
            if random.random() < op_mutation_prob:
                available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
                new_ops[ind] = random.choice(available)

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def run_evolution_search(nasbench,
                         max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []  # (validation, spec) tuples

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    for _ in range(population_size):
        spec = random_spec()
        data = nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break

    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        new_spec = mutate_spec(best_spec, mutation_rate)

        data = nasbench.query(new_spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests


def run_random_search(nasbench, max_time_budget=5e6):
    """Run a single roll-out of random search to a fixed time budget."""
    nasbench.reset_budget_counters()
    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    while True:
        spec = random_spec()
        data = nasbench.query(spec)

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        if time_spent > max_time_budget:
            # Break the first time we exceed the budget.
            break

    return times, best_valids, best_tests


def manual_define_sampled_search(visualize=False):
    with open('./data/nasbench_hash-rank_v7_e9_op3.json', 'r') as f:
        new_dict = json.load(f)

    # Test sampling the data
    # Not random, but just take a bunch of from
    # sample a set of 30 hash, build their graph and train.
    indices = [10, 1000, 2000, 4000, 5000]
    for i in range(1, 5):
        indices += [i for i in range(int(1e5 * i), int(1e5 * i + 5))]

    indices += [i for i in range(423000, 423005)]
    # Plot the histogram.
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(4, 5))
        ax.hist([new_dict['validation_accuracy'][k] for k in indices], bins=20)
        fig.show()
        print(indices)
        print(len(indices))

    hashs = ''
    rank = ''
    for k in indices:
        hashs += f'\"{new_dict["hash"][str(k)]}\" '
        rank += f'\"{k}\" '
    return hashs, rank


def obtain_full_model_spec(num_vertices, op=CONV3X3):
    full_connection = 1 - np.tril(np.ones([num_vertices, num_vertices]))
    ops = [INPUT,] + (num_vertices - 2) * [op,] + [OUTPUT]
    return ModelSpec_v2(matrix=full_connection, ops=ops)


def obtain_random_spec(num_vetices):
    while True:
        matrix = np.random.choice(ALLOWED_EDGES, size=(num_vetices, num_vetices))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(ALLOWED_OPS, size=(num_vetices)).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT

        spec = ModelSpec_v2(matrix=matrix, ops=ops)
        if spec.valid_spec:
            if not len(spec.ops) > 2:
                continue
            break
    # IPython.embed()
    return spec
