import argparse
from math import exp

parser = argparse.ArgumentParser("cifar")

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--init_channels', type=int, default=20, help='num of init channels')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--keep_prob', type=float, default=0.9, help='dropout')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# for FBNet, only change is from Softmax to Gumbelsoftmax, to get a weights more closer to [0,1]
parser.add_argument('--gumble_softmax_temp',type=float, default=5.0, help='softmax temperature for Gumbel Softmax')
parser.add_argument('--gumble_softmax_decay', type=float, default=exp(-0.045), help='Decay for each epoch.')
dartsargs = parser.parse_args("")
