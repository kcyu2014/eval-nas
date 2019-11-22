import argparse

from utils import get_logger

logger = get_logger()

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Catastrophic forgetting settings
cata_arg = add_argument_group('catastrophic')
cata_arg.add_argument('--start_using_fisher', type=int, default=0)
cata_arg.add_argument('--num_batch_per_iter', type=int, default=10)
cata_arg.add_argument('--set_fisher_zero_per_iter', type=int, default=-1)
cata_arg.add_argument('--lambda_fisher', type=float, default=0.9, help='momentum to update fisher')
cata_arg.add_argument('--fisher_clip_by_norm', type=float, default=0.0, help='Clip fisher before it is too large')
cata_arg.add_argument('--alpha_fisher', type=float, default=1.0, help='increase penalty')
cata_arg.add_argument('--momentum', type=str2bool, default=True, help='use momentum')

cata_arg.add_argument('--alpha_decay', type=float, default=2)
cata_arg.add_argument('--alpha_decay_after', type=float, default=15)

# train shared_model
cata_arg.add_argument('--shared_valid_fisher', type=str2bool, default=True)
cata_arg.add_argument('--shared_ce_fisher', type=str2bool, default=True)

# train controller
cata_arg.add_argument('--controller_update_fisher', type=str2bool, default=False)
cata_arg.add_argument('--train_controller', type=str2bool, default=False)
cata_arg.add_argument('--stop_training_controller', type=int, default=1000)
cata_arg.add_argument('--nb_batch_reward_controller', type=int, default=20, help='Number of batches to calculate the reward for the controller')


# Evaluation (set to 100 if not used)
cata_arg.add_argument('--start_evaluate_diff', type=int, default=0)


# CNN search argument
cnn_arg = add_argument_group('cnn')
cnn_arg.add_argument('--search_mode', type=str, choices=['macro', 'micro'], default='micro')
cnn_arg.add_argument('--output_classes', type=int, default=10)
cnn_arg.add_argument('--cnn_height', type=int, default=32)
cnn_arg.add_argument('--cnn_width', type=int, default=32)
cnn_arg.add_argument('--cnn_input_channels', type=int, default=3)
cnn_arg.add_argument('--cnn_final_filter_size', type=int, default=768 // 2)
cnn_arg.add_argument('--cnn_num_repeat_normal', type=int, default=6)
cnn_arg.add_argument('--cnn_num_modules', type=int, default=3)
cnn_arg.add_argument('--num_workers', type=int, default=4)

##########################################################
#            ORIGINAL CONFIG                             #
##########################################################
# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['rnn', 'micro_cnn', 'marco_cnn'], default='rnn')


# Shared parameters for PTB
net_arg.add_argument('--shared_wdrop', type=float, default=0.5)
net_arg.add_argument('--shared_dropout', type=float, default=0.4)
net_arg.add_argument('--shared_dropoute', type=float, default=0.1)
net_arg.add_argument('--shared_dropouti', type=float, default=0.65)
net_arg.add_argument('--shared_embed', type=int, default=1000)
net_arg.add_argument('--shared_hid', type=int, default=1000)
net_arg.add_argument('--shared_rnn_max_length', type=int, default=35)
net_arg.add_argument('--shared_rnn_activations', type=eval,
                     default="['tanh', 'ReLU', 'identity', 'sigmoid']")


# Shared parameters for CIFAR
net_arg.add_argument('--cnn_hid', type=int, default=64)


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ptb')


# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test'],
                       help='train: Training ENAS, derive: Deriving Architectures')
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=310)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])


# Controller
net_arg.add_argument('--num_blocks', type=int, default=12)
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--controller_hid', type=int, default=100)
learn_arg.add_argument('--policy_batch_size', type=int, default=5)
learn_arg.add_argument('--ppl_square', type=str2bool, default=False)
learn_arg.add_argument('--reward_c', type=int, default=1,
                       help="80 for RNN, 1 for CNN")
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95)
learn_arg.add_argument('--discount', type=float, default=1.0)
learn_arg.add_argument('--controller_max_step', type=int, default=1000,
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_lr_cosine', type=str2bool, default=False)
learn_arg.add_argument('--controller_lr_max', type=float, default=0.05,
                       help="lr max for cosine schedule")
learn_arg.add_argument('--controller_lr_min', type=float, default=0.001,
                       help="lr min for cosine schedule")
learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=1e-4, help='Entropy weight')

# Shared parameters
learn_arg.add_argument('--shared_initial_step', type=int, default=0)
learn_arg.add_argument('--shared_max_step', type=int, default=50,
                       help='step for shared parameters')
learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_optim', type=str, default='sgd')
learn_arg.add_argument('--shared_lr', type=float, default=20.0)
learn_arg.add_argument('--shared_decay', type=float, default=0.96)
learn_arg.add_argument('--shared_decay_after', type=float, default=15)
learn_arg.add_argument('--shared_l2_reg', type=float, default=1e-7)
learn_arg.add_argument('--shared_grad_clip', type=float, default=0.25)

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=5)
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)
misc_arg.add_argument('--comment', type=str, default=None, help='Use to discriminate the train folder')


def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed


enasargs = get_args()

if __name__ == '__main__':
    args, unparsed = get_args()
    print(args.__dict__)