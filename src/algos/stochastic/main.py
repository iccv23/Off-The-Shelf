import sys
import numpy as np
import torch


import os
if os.environ["PWD"].startswith("/ai/base/"):
    sys.path.append('/ai/base/G-FIR/')
else:
    sys.path.append('/home/user/data/codes/G-FIR/')
# user defined
from trainer import Trainer
from src.options.options import Options


def main(args):

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('\nDevice:{}'.format(device))

    trainer = Trainer(args)
    trainer.do_training()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)
