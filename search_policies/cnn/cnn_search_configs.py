import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser("cifar")

# common space

parser.add_argument('--dataset', type=str, default='cifar10', help="dataset to use.")
parser.add_argument('--num_intermediate_nodes', type=int, default=2, help='NASbench cell vertex')
parser.add_argument('--search_policy', type=str, default='random', help="Search Policy")
parser.add_argument('--search_space', type=str, default='nasbench', help="Search Space to use")
parser.add_argument('--seed_range_start', type=int, default=1268)
parser.add_argument('--seed_range_end', type=int, default=1269)
parser.add_argument('--evaluation_seed', type=int, default=1278, help='default evaluation seeds.')
parser.add_argument('--supernet_train_method', type=str, default='darts', help='Training method for supernet model')
parser.add_argument('--continue_train', action='store_true', help='continue train from a checkpoint')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--evaluate_after_search', default=False, action='store_true')
parser.add_argument('--tensorboard', default=True, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
# parser.add_argument('--no-tensorbard', dest='tensorboard', action='store_false')
# parser.set_defaults(tensorboard=True)

# search_main configs
parser.add_argument('--save_every_epoch', type=int, default=30, help='evaluate and save every x epochs.')
parser.add_argument('--extensive_save', type=str2bool, default=True,
                    help='Extensive evaluate 3 consequtive epochs for a stable results.')
parser.add_argument('--main_path', type=str, default=None, help='path to save EXPs.')
parser.add_argument('--resume_path', type=str, default=None, help='path to save the final model')
parser.add_argument('--test_dir', type=str, default='none', help='path to save the tests directories')
parser.add_argument('--tboard_dir', type=str, default='runs', help='path to save the tboard logs')

# DARTS based configs.
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--evaluate_batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpus', type=int, default=1, help='GPU count')
parser.add_argument('--epochs', type=int, default=151, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# For customization, NASBench training procedure.
parser.add_argument("--model_spec_hash", type=str, default=None, help='Model spec hashing for the NASBench. '
                                                                      'OR index for NAO-DARTS search space.')
parser.add_argument('--nasbench_module_vertices', type=int, default=5, help='Maximum number of vertex.')
parser.add_argument('--num_archs_subspace', type=int, default=10, help='Number of sub-space architectures to sample.')
parser.add_argument('--controller_random_arch', type=int, default=20)
# Settings for MixedVertex Op in nas_bench.operations
parser.add_argument('--nasbenchnet_vertex_type', type=str, default='mixedvertex')
parser.add_argument('--channel_dropout_method', type=str, default='fixed_chunck',
                    help='Channel dropout configs. Used in '
                         'nas_bench.operations.ChannelDropout')
parser.add_argument('--channel_dropout_dropouto', type=float, default=0.0,
                    help='additional dropout operation after the Channel dropout'
                    )
parser.add_argument('--dynamic_conv_method', type=str, default='fixed_chunck',
                    help='Dynamic conv is channel-droupout operate on the convolutional kernels.'
                         'All methods are the same as channel dropout method.')
parser.add_argument('--dynamic_conv_dropoutw', type=float, default=0.0,
                    help='additional kernel dropout.')
parser.add_argument('--wsbn_train', type=str2bool, default=True, help="nasbench enable bn during training or not.")
parser.add_argument('--wsbn_sync', type=str2bool, default=False, help="Enable Sync BN in NASBench MixedOp.")
parser.add_argument('--wsbn_track_stat', type=str2bool, default=True, help="Disable BN tracking status.")


def build_default_args(parse_str=''):
    default_args = parser.parse_args(parse_str)
    return default_args
