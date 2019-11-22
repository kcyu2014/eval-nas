"""
This is a cnn search policy wrapper.
"""

import logging

from search_policies.cnn.darts_policy.darts_official_configs import darts_official_args
from search_policies.cnn.enas_policy.enas_micro.train_search import get_enas_microcnn_parser
from utils import DictAttr
from nasbench.lib.config import build_config as nasbench_build_config

# sampler based policies.
from .darts_policy.darts_search_policy import DARTSNasBenchSearch, DARTSMicroCNNSearchPolicy
from .darts_policy.fbnet_search_policy import FBNetNasBenchSearch
from .enas_policy.enas_search_policy import ENASNasBenchSearch, ENASMicroCNNSearchPolicy, ENASNasBenchGroundtruthPolicy
from .nao_policy.nao_search_policy import NAONasBenchSearch, NAONasBenchGroundTruthPolicy
# One-shot based policies.
from .random_policy.nasbench_weight_sharing_policy import NasBenchWeightSharingPolicy, NasBenchNetOneShotPolicy
from .random_policy.nasbench_weight_sharing_fairnas_policy import NasBenchNetTopoFairNASPolicy
# nasbench arguments
from .enas_policy.configs import enasargs
from .nao_policy.configs import naoargs
from .darts_policy.configs import dartsargs


def wrap_config(config, prefix, keys, args):
    logging.info("Wrapping the keys into search_policy_configs")
    for k in keys:
        if prefix == '':
            v = getattr(args, f'{k}')
        else:
            v = getattr(args, f'{prefix}_{k}')
        logging.info(f"Setting config key {k} to {v}")
        setattr(config, k, v)


class CNNCellSearchPolicyFactory:

    @staticmethod
    def factory(args):
        if 'nasbench' in args.search_space:
            # wrapping nasbench_config.
            nasbench_search_config = DictAttr(nasbench_build_config())
            wrap_config(nasbench_search_config, 'nasbench', args=args,
                        keys=[
                            'module_vertices',
                            # add more if necessary.
                        ])
            args.nasbench_config = nasbench_search_config

            if args.search_policy == 'ws_random':
                return NasBenchWeightSharingPolicy(args)
            elif args.search_policy in ['fairnas', 'spos', 'oneshot']:
                return NasBenchNetOneShotPolicy(args)
            elif args.search_policy == 'topo_fairnas':
                return NasBenchNetTopoFairNASPolicy(args)
            elif args.search_policy in ['nao', 'nao-groundtruth']:

                """
                The experiment settting here is purely for NASBench experiments, 
                rerun in original space, need to do more operations.
                """
                nao_search_config = DictAttr(naoargs.__dict__)
                nodes = args.num_intermediate_nodes
                # setting embedding size proportional to nodes.
                base_embed_size = 12 * (nodes - 1)
                nao_search_config.controller_encoder_hidden_size = base_embed_size
                nao_search_config.controller_decoder_hidden_size = base_embed_size
                nao_search_config.controller_encoder_emb_size = base_embed_size
                nao_search_config.controller_decoder_emb_size = base_embed_size
                # setting vocabulary size.
                nao_search_config.controller_encoder_vocab_size = nodes + 3 + 1
                nao_search_config.controller_decoder_vocab_size = nodes + 3 + 1
                nao_search_config.child_nodes = nodes
                nao_search_config.controller_source_length = nodes * 2
                nao_search_config.controller_encoder_length = nodes * 2
                nao_search_config.controller_decoder_length = nodes * 2

                if args.debug:
                    # reduce time
                    nao_search_config.child_epochs = 8
                    nao_search_config.controller_seed_arch = 10
                    nao_search_config.controller_random_arch = 10
                    nao_search_config.controller_new_arch = 10
                    nao_search_config.controller_epochs = 4
                    nao_search_config.child_eval_epochs = "1"

                args.nao_search_config = nao_search_config
                if args.search_policy == 'nao':
                    return NAONasBenchSearch(args)
                elif args.search_policy == 'nao-groundtruth':
                    return NAONasBenchGroundTruthPolicy(args)

            elif args.search_policy in ['enas', 'enas-groundtruth']:
                enas_search_config = DictAttr(enasargs.__dict__)
                enas_search_config.child_num_cells = args.num_intermediate_nodes
                args.enas_search_config = enas_search_config
                if args.search_policy == 'enas':
                    return ENASNasBenchSearch(args)
                elif args.search_policy == 'enas-groundtruth':
                    return ENASNasBenchGroundtruthPolicy(args)
            elif args.search_policy == 'darts':
                darts_search_config = DictAttr(dartsargs.__dict__)
                darts_search_config.layers = args.num_intermediate_nodes
                args.darts_search_config = darts_search_config
                return DARTSNasBenchSearch(args)

            elif args.search_policy == 'fbnet':
                # follow the original paper settings.
                # do not use second order gradient.
                fbnet_search_config = DictAttr(dartsargs.__dict__)
                fbnet_search_config.layers = args.num_intermediate_nodes
                fbnet_search_config.train_portion = 0.8
                fbnet_search_config.unrolled = False
                fbnet_search_config.momentum = 0.9
                fbnet_search_config.learning_rate = 0.1
                fbnet_search_config.arch_learning_rate = 0.01
                fbnet_search_config.arch_weight_decay = 5e-4
                args.fbnet_search_config = fbnet_search_config
                return FBNetNasBenchSearch(args)
        elif args.search_space == 'original':
            """
            Experiment here should be done with the following setting:
                Each policy has its own model
                    Run their original search on it for 10 times, to get 10 best models
                    WS-Random to get another 10 random models
                in total, 3 * (10 + 10) = 60 training from scratch, in each of space
                ~ 180 GPU days + exp times. 
            """
            # intermediate node = 5 in this cases.
            if args.search_policy == 'enas':
                # get default parser
                enas_args = get_enas_microcnn_parser().parse_args('')
                wrap_config(enas_args, '',
                            keys=['seed', 'epochs', 'data'], args=args)
                return ENASMicroCNNSearchPolicy(args, enas_args)
            elif args.search_policy == 'darts':
                steps = 4 # this is fixed
                darts_config = darts_official_args
                wrap_config(darts_config, '', keys=['seed', 'epochs', 'data'], args=args)
                return DARTSMicroCNNSearchPolicy(args, darts_config)
            elif args.search_policy == 'nao':
                raise NotImplementedError("TODO This is not yet supported. "
                                          "please run the official code for 10 times, "
                                          "and parse the data manually.")
        else:
            raise NotImplementedError("not supported")
