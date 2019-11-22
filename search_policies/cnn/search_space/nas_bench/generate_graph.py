from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import sys
from argparse import Namespace

from nasbench.lib import graph_util
import numpy as np
import logging


def generate_graph(max_vertices, max_edges, num_ops, verify_isomorphism, output_file):

    FLAGS = Namespace(max_vertices=max_vertices,
                      num_ops=num_ops,
                      max_edges=max_edges,
                      verify_isomorphism=verify_isomorphism,
                      output_file=output_file
                      )

    total_graphs = 0    # Total number of graphs (including isomorphisms)
    # hash --> (matrix, label) for the canonical graph associated with each hash
    buckets = {}

    logging.info('Using %d vertices, %d op labels, max %d edges',
                 FLAGS.max_vertices, FLAGS.num_ops, FLAGS.max_edges)
    for vertices in range(2, FLAGS.max_vertices+1):
        for bits in range(2 ** (vertices * (vertices-1) // 2)):
            # Construct adj matrix from bit string
            matrix = np.fromfunction(graph_util.gen_is_edge_fn(bits),
                                     (vertices, vertices),
                                     dtype=np.int8)

            # Discard any graphs which can be pruned or exceed constraints
            if (not graph_util.is_full_dag(matrix) or
                    graph_util.num_edges(matrix) > FLAGS.max_edges):
                continue

            # Iterate through all possible labelings
            for labeling in itertools.product(*[range(FLAGS.num_ops)
                                                for _ in range(vertices-2)]):
                total_graphs += 1
                labeling = [-1] + list(labeling) + [-2]
                fingerprint = graph_util.hash_module(matrix, labeling)

                if fingerprint not in buckets:
                    buckets[fingerprint] = (matrix.tolist(), labeling)

                # This catches the "false positive" case of two models which are not
                # isomorphic hashing to the same bucket.
                elif FLAGS.verify_isomorphism:
                    canonical_graph = buckets[fingerprint]
                    if not graph_util.is_isomorphic(
                            (matrix.tolist(), labeling), canonical_graph):
                        logging.fatal('Matrix:\n%s\nLabel: %s\nis not isomorphic to'
                                      ' canonical matrix:\n%s\nLabel: %s',
                                      str(matrix), str(labeling),
                                      str(canonical_graph[0]),
                                      str(canonical_graph[1]))
                        sys.exit()

        logging.info('Up to %d vertices: %d graphs (%d without hashing)',
                     vertices, len(buckets), total_graphs)

    with open(FLAGS.output_file, 'w') as f:
        json.dump(buckets, f, sort_keys=True)
