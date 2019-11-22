import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--ratio', type=float, default=0.9, help='ratio of data')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0)

# Child settings
parser.add_argument('--child_num_branches', type=int, default=3)
parser.add_argument('--child_num_cells', type=int, default=5, help='==num_intermediate_node')
parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_num_layers', type=int, default=6, help='num layers in total, each layer is a search cell')
parser.add_argument('--child_out_filters', type=int, default=20)
parser.add_argument('--child_eval_batch_size', type=int, default=64)
parser.add_argument('--child_use_aux_heads', type=bool, default=False)
parser.add_argument('--child_l2_reg', type=float, default=0.00025)
parser.add_argument('--child_grad_bound', type=float, default=5.0)

# controller setting
parser.add_argument('--controller_lr', type=float, default=0.0035)
parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
parser.add_argument('--controller_train_steps', type=int, default=30)
parser.add_argument('--lstm_size', type=int, default=64)
parser.add_argument('--lstm_num_layers', type=int, default=1)
parser.add_argument('--lstm_keep_prob', type=float, default=0)
parser.add_argument('--temperature', type=float, default=5.0)
parser.add_argument('--entropy_weight', type=float, default=0.0001)
parser.add_argument('--bl_dec', type=float, default=0.99)

# Adapted.
parser.add_argument('--controller_train_every', type=int, default=1)
parser.add_argument('--controller_random_arch', type=int, default=10)
parser.add_argument('--controller_new_arch', type=int, default=10)
parser.add_argument('--controller_num_aggregate', type=int, default=10)
parser.add_argument('--controller_sample_batch', type=int, default=10)

# NEW ARGUMENTS NEED TO BE IMPLEMENTED
parser.add_argument('--child_keep_prob', type=float, default=0.9, help="MAKE THIS HAPPENS")
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.6)


enasargs = parser.parse_args("")