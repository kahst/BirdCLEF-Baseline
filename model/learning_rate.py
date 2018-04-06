# This file contains learning rate schedules for training.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import math
import numpy as np

import config as cfg

def dynamicLearningRate(mode, epoch):

    lr_start = cfg.LEARNING_RATE['start']
    lr_end = cfg.LEARNING_RATE['end']

    if mode == 'linear':

        # Linear interpolation
        lr = lr_start - (epoch - 1) * ((lr_start - lr_end) / max(1, (cfg.EPOCHS - 1)))
        
    elif mode.split('-')[0] == 'step':

        # Steps after certain amount of epochs
        stepsize = int(mode.split('-')[1])
        steps = cfg.EPOCHS // stepsize - 1
        stepdiff = (lr_start - lr_end) / steps
        lr = lr_start - ((epoch - 1) // stepsize * stepdiff)

    elif mode == 'cosine':

        # Cosine annealing schedule used in snapshot ensembles
        n_snapshots = 1
        cos_inner = np.pi * ((epoch - 1) % (cfg.EPOCHS // n_snapshots))
        cos_inner /= (cfg.EPOCHS) // n_snapshots
        cos_out = np.cos(cos_inner) + 1
        lr = float(lr_start / 2 * cos_out)

    elif mode == 'root' and epoch > 0:

        # Square root scheduling
        lr = lr_start / math.sqrt(epoch)
        
    else:

        # Constant LR
        lr = lr_start

    return lr

if __name__ == '__main__':

    for epoch in range(1, cfg.EPOCHS + 1):

        print dynamicLearningRate('cosine', epoch)
