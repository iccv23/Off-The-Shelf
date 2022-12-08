"""
    Parse input arguments
"""
import argparse
import numpy as np


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Options for G-FIR/ZS-SBIR')

        import os
        if os.environ["PWD"].startswith("/ai/base/"):
            parser.add_argument('-root', '--root_path', default='/ai/base/data/', type=str)
        else:
            parser.add_argument('-root', '--root_path', default='/home/user/data/datasets/', type=str)
        parser.add_argument('-path_cp', '--checkpoint_path', default='./saved_models/', type=str)
        parser.add_argument('-resume', '--resume_dict', default=None, type=str, help='checkpoint file to resume training from')

        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['Sketchy', 'DomainNet', 'TUBerlin'])
        parser.add_argument('-eccv', '--is_eccv_split', choices=[0, 1], default=1, type=int, help='whether or not to use eccv18 split\
                            if dataset="Sketchy"')

        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='sketch', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting', 'real'])
        parser.add_argument('-hd', '--holdout_domain', default='quickdraw', choices=['quickdraw', 'clipart', 'infograph', 'sketch', 'painting', 'real'])
        parser.add_argument('-gd', '--gallery_domain', default='real', choices=['clipart', 'infograph', 'photo', 'painting', 'real', 'sketch'])
        parser.add_argument('-aux', '--include_auxillary_domains', choices=[0, 1], default=1, type=int, help='whether(1) or not(0) to include\
                            domains other than seen domain and gallery')

        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')

        # Loss weight & reg. parameters
        parser.add_argument('-nc', '--num_codebooks', default=2, type=int, help='Number of codebooks (M)')
        parser.add_argument('-nk', '--codebook_size', default=256, type=int, help='Codebook size (K)')
        parser.add_argument('--mode', type=str, default='simple', help="Loss mode of pqnet learning. Options: 'ce', 'triplet'.")
        parser.add_argument('--hp_lambda', type=float, default=1, help='Weight for entropy regularization loss.')
        parser.add_argument('--hp_gamma', type=float, default=0.5, help='Weight for codebook regularization loss.')
        parser.add_argument('-alpha', '--alpha', default=10, type=float, help='Alpha scaling parameter for soft codeword assignment.')
        parser.add_argument('-l2', '--l2_reg', default=0.0, type=float, help='L2 Weight Decay for optimizer')

        parser.add_argument('--queue_begin_epoch', type=int, default=np.inf, help='The epoch for starting using memory queue.')

        parser.add_argument('--gpus', type=int, default=4, help='GPUs used for DDP.')

        # Size parameters
        parser.add_argument('-feat', '--feat_dim', default=32, type=int, help='output feature dim (default = M * 16)')
        parser.add_argument('-imsz', '--image_size', default=224, type=int, help='Input size for query/gallery domain sample')

        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=128, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=6, help='Number of workers in data loader')

        # Optimization parameters
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('--start_lr', type=float, default=1e-5, help='Learning rate at the start of warmup.')
        parser.add_argument('--final_lr', type=float, default=1e-5, help='Final learning rate of cosine decaying schedule.')
        parser.add_argument('-lr', '--lr', type=float, default=0.001, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('--warmup_epoch_num', type=int, default=1, help='Number of warmup epochs for lr scheduler.')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        parser.add_argument('--disable_scheduler', dest='use_scheduler', action='store_false', help='Disabling the learning rate scheduler.')
        parser.set_defaults(use_scheduler=True)

        # Checkpoint parameters
        parser.add_argument('-es', '--early_stop', type=int, default=30, help='Early stopping epochs.')

        # I/O parameters
        parser.add_argument('-log', '--log_interval', type=int, default=400, metavar='N', help='How many batches to wait before logging training status')

        self.parser = parser


    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
