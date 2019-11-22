"""
NASBench only support ModelSpec -> Hash.

What I need to do is to extend such dataset, to support Hash -> ModelSpec.
DO IT QUICK...
It is because I need to sample a data according to
"""
import copy
import logging
import os
import json

import IPython
import numpy as np

import utils
from nasbench.api import NASBench, OutOfDomainError
from nasbench.api import ModelSpec

AVAILABLE_OPS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
AVAILABLE_OPS_ALL = ['output', 'input'] + AVAILABLE_OPS
from .generate_graph import generate_graph


class ModelSpec_v2(ModelSpec):
    def model_spec_to_json(self):
        data = [self.matrix.tolist(),
                [AVAILABLE_OPS_ALL.index(a) - 2 for a in self.ops]]
        return data

    @classmethod
    def load_from_list(cls, lists, available_ops=AVAILABLE_OPS):
        assert len(lists) == 2, 'List data must contain list[0] as adjacency matrix, list[1], as op-ids' \
                                'got {} instead'.format(lists)
        available_ops = ['output', 'input', ] + list(available_ops)
        op_list = [available_ops[i + 2] for i in lists[1]]
        return cls(lists[0], op_list)

    def __str__(self):
        return 'Adjacent matrix: {} \n' \
               'Ops: {}\n'.format(self.matrix.tolist(), self.ops)

    def hash_spec(self, canonical_ops=AVAILABLE_OPS):
        return super(ModelSpec_v2, self).hash_spec(canonical_ops)

    def new(self):
        from copy import copy
        return ModelSpec_v2(self.matrix.copy(), copy(self.ops))

    def resume_original(self):
        """
        For NASbench with supernet training, we need to keep topology the
        same while keeping the hash sepc correct.
        :return:
        """
        # These two used to compute hash.
        self.matrix_for_hash = self.matrix
        self.ops_for_hash = self.ops

        self.matrix = copy.deepcopy(self.original_matrix)
        self.ops = copy.deepcopy(self.original_ops)


class NASBench_v2(NASBench):
    """
    This new version NAS Bench has additional API: for the following usage:

        Table 1. NASBench v1 Dataset
            Hash -> Performance and all detailed information

        Table 2.
            Hash -> ModelSpec (raw format)
            ModelSpec (Raw): [model_graph, op_list]
                - model_graph: defined just as [[adjacent matrix]]
                - op_list: [-1, ...(0,1,2), -2]
                    -1: 'input'
                    -2: 'output'
                    0 : 'conv3x3-bn-relu'
                    1 : 'conv1x1-bn-relu'
                    2 : 'maxpool3x3'
    """
    available_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

    def __init__(self, dataset_file, model_arch_file=None, model_perf_file=None, seed=None, only_hash=True, config='v7_e9_op3'):
        loaded_datast = False
        dataset_dir = os.path.dirname(os.path.realpath(dataset_file))
        config_split = config.split('_')
        vertices, maxedge, ops = eval(config_split[0][1:]),eval(config_split[1][1:]), eval(config_split[2][2:])

        logging.info('NASBench dataset v2. Max vertices {}, edge {}, ops {}'.format(vertices, maxedge, ops))
        self.only_hash = only_hash
        if not os.path.exists(dataset_file):
            print("dataset file ", dataset_file)
            print('current dir ', os.path.dirname(os.path.realpath(__file__)))
            print('path to look at', os.path.realpath(dataset_file))
        # IPython.embed()
        all_graph_file = os.path.join(dataset_dir, f'nasbench_all_graphs_{config}.json')
        model_arch_file = model_arch_file or os.path.join(dataset_dir, f'nasbench_hash_rank_bytest_{config.replace("_", "-")}.json')
        model_perf_file = model_perf_file or os.path.join(dataset_dir, f'nasbench_perfs_bytest_{config.replace("_", "-")}.json')

        if not os.path.exists(all_graph_file):
            print("Trying to make all graph file {}".format(all_graph_file))
            config_split = config.split('_')
            generate_graph(eval(config_split[0][1:]),
                           eval(config_split[1][1:]),
                           eval(config_split[2][2:]),
                           True,
                           all_graph_file)

        with open(all_graph_file, 'r') as f:
            self.hash_dict = json.load(f)

        if not os.path.exists(model_perf_file) or not os.path.exists(model_arch_file):
            if not loaded_datast:
                print("Try to make perf file ")
                super(NASBench_v2, self).__init__(dataset_file, seed)
                self.only_hash = False
                loaded_datast = True
            self._make_rank_hash_perf_file(model_perf_file, config)

        with open(model_perf_file, 'r') as f:
            self.perf_rank = json.load(f)
            self.perf_rank_name = ['validation_accuracy', 'test_accuracy']

        with open(model_arch_file, 'r') as f:
            self.hash_rank = json.load(f)

        if not only_hash and not loaded_datast:
            super(NASBench_v2, self).__init__(dataset_file, seed)

    def _make_rank_hash_perf_file(self, f_name, config=f'v7_e9_op3'):

        def make_perfs_file(perf_filename):
            """ move to nasbench_v2 """
            hash_perfs = {}
            for ind, hash in enumerate(self.hash_dict.keys()):
                perfs = self.query_hash(hash)
                hash_perfs[hash] = (perfs['validation_accuracy'], perfs['test_accuracy'])
            utils.save_json(hash_perfs, perf_filename)
            return hash_perfs

        filename = f'data/nasbench/nasbench_perfs_{config}.json'
        if not os.path.exists(filename):
            hash_perfs = make_perfs_file(filename)
        else:
            hash_perfs = utils.load_json(filename)
        self.hash_perfs_keys = ('validation_accuracy', 'test_accuracy')

        hashs = list(self.hash_dict.keys())
        f_config = config.replace('_', '-')
        hash_rank_filename = f'nasbench_hash_rank_bytest_{f_config}.json'
        perf_filename = f'nasbench_perfs_bytest_{f_config}.json'

        perfs = [hash_perfs[h] for h in hashs]
        t_ps = [p[1] for p in perfs]
        print("sorting the hashs by testing accuracy.")
        sorted_hash_indices = np.argsort(t_ps)
        s_hashs = [hashs[i] for i in sorted_hash_indices]
        s_perfs = [perfs[i] for i in sorted_hash_indices]
        utils.save_json(s_perfs, 'data/nasbench/' + perf_filename)
        utils.save_json(s_hashs, 'data/nasbench/' + hash_rank_filename)

    def hash_to_model_spec(self, hash):
        return ModelSpec_v2.load_from_list(self.hash_dict[hash])

    def query_hash(self, hash, **kwargs):
        return self.query(self.hash_to_model_spec(hash), **kwargs)

    def query(self, model_spec, epochs=108, stop_halfway=False, average=True):
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).
          average: Return averaged value. If False, use NASBench original api.

        Returns:
          dict containing the evaluated data for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if not average:
            return super(NASBench_v2, self).query(model_spec, epochs, stop_halfway)

        if epochs not in self.valid_epochs:
            raise OutOfDomainError('invalid number of epochs, must be one of %s'
                                   % self.valid_epochs)

        fixed_stat, computed_stats = self.get_metrics_from_spec(model_spec)
        # print(fixed_stat)
        # print(computed_stats)

        num_repeat = self.config['num_repeats']
        sampled_index = range(0, num_repeat)

        def _query_based_on_sample_index(_sampled_index, computed_stats):
            computed_stat = computed_stats[epochs][_sampled_index]
            # print(computed_stat)

            data = {}
            data['module_adjacency'] = fixed_stat['module_adjacency']
            data['module_operations'] = fixed_stat['module_operations']
            data['trainable_parameters'] = fixed_stat['trainable_parameters']

            if stop_halfway:
                data['training_time'] = computed_stat['halfway_training_time']
                data['train_accuracy'] = computed_stat['halfway_train_accuracy']
                data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
                data['test_accuracy'] = computed_stat['halfway_test_accuracy']
            else:
                data['training_time'] = computed_stat['final_training_time']
                data['train_accuracy'] = computed_stat['final_train_accuracy']
                data['validation_accuracy'] = computed_stat['final_validation_accuracy']
                data['test_accuracy'] = computed_stat['final_test_accuracy']

            self.training_time_spent += data['training_time']
            if stop_halfway:
                self.total_epochs_spent += epochs // 2
            else:
                self.total_epochs_spent += epochs
            return data

        datas = [ _query_based_on_sample_index(ind, computed_stats) for ind in sampled_index]

        final_data = {}
        for run_id, data in enumerate(datas):
            for k in data.keys():
                if any(keyword in k for keyword in ['module', 'parameters']):
                    final_data[k] = data[k]
                else:
                    final_data[k] = data[k] / num_repeat if k not in final_data.keys() else \
                        final_data[k] + data[k] / num_repeat
                    final_data[f'{k}_run-{run_id}'] = data[k]

        for k in ['train_accuracy', 'validation_accuracy', 'test_accuracy', 'training_time']:
            final_data[k + '_std'] = np.std(np.array([d[k] for d in datas], dtype=np.float32))
        return final_data

    def model_hash_rank(self, full_spec=False):
        """
        Return model's hash according to its rank in given search space.
        Output:
            if not full_spec:   list[hash_1, hash_2 ...]
            if full_spec:       list[hash_1, hash_2 ...], list[model_spec_1, model_spec_2 ...]

        :param full_spec: if True, return the spec of each model architecture.
        :return:
        """
        logging.info('return the entire model hash rank. '
                     'Within current searching space, the number of total models are \n'
                     f'Total models: {len(self.hash_rank)}. ')
        # IPython.embed(header='model_hash_rank')
        if full_spec:
            model_specs = []
            for ind, h in enumerate(self.hash_rank):
                model_specs.append(self.hash_to_model_spec(h))
            return self.hash_rank, model_specs

        return self.hash_rank

    # TODO implement, hash_perfs, perfs_rank
    def __repr__(self):
        # built the better visualization.
        string = f'NASBench v2 dataset( \n' \
                 f'\t Total number:\t{len(self.model_hash_rank())}' \
            f')'
        return string

