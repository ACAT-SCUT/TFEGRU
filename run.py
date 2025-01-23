import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Time-Frequency Enhanced Gated Recurrent Unit with Attention')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--model', type=str, default='TFEGRU',help='model name')
parser.add_argument('--save_data', type=bool, default=True, help='save predict result or not')

# data loader
parser.add_argument('--root_path', type=str, default='./data/google/', help='root path of the data file')
parser.add_argument('--dataset', type=str, default='google', help='name of dataset')
parser.add_argument('--result_folder', type=str, default='./results/', help='saving path of the result')
parser.add_argument('--counts', nargs='+', type=int, help='counts of workloads')
parser.add_argument('--train_size', type=float, default=0.7, help='size of training set')
parser.add_argument('--val_size', type=float, default=0.2, help='size of validating set')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# Model
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--input_size', type=int, default=100, help='input size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--patience', type=int, default=5, help='early stopping')


# optimization
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--decay_rate', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--num_epochs', type=int, default=30, help='train epochs')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

os.makedirs(args.result_folder, exist_ok=True)

exp = Exp(args)

exp.train_and_evaluate()
torch.cuda.empty_cache()
