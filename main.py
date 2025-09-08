import os
import random
import argparse
import configparser

import numpy as np
from torch.backends import cudnn
import torch

from solver import Solver
import sys



class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    else:
        solver.train()
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)

    # Data arguments
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=55)
    parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--data_path', type=str, default='dataset/MSL')

    # Parameter arguments
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--fr', type=float, default=0.4)
    parser.add_argument('--tr', type=float, default=0.4)
    parser.add_argument('--seq_size', type=int, default=5)

    # Model arguments
    parser.add_argument('--describe', type=str, default='none')
    parser.add_argument('--mode', type=str, choices=['train', 'test','all'], default='train', help="Set the mode (train/test)")
    parser.add_argument('--model_save_path', type=str, default='cpt', help="Path to save model")

    config = parser.parse_args()

    args = vars(config)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    sys.stdout = Logger("result/"+ config.dataset +".log", sys.stdout)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
