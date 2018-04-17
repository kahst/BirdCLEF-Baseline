# This file contains a short example of the evaluation process
# including training and testing.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import numpy as np

import config as cfg
from model import lasagne_net as birdnet
from model import lasagne_io as io
from utils import stats
from utils import log
import train
import test

###################### EVALUATION #######################
def evaluate():

    # Clear stats
    stats.clearStats(True)

    # Parse Dataset
    cfg.CLASSES, TRAIN, VAL = train.parseDataset()

    # Build Model
    NET = birdnet.build_model()

    # Train and return best net
    best_net = train.train(NET, TRAIN, VAL)

    # Load trained net
    SNAPSHOT = io.loadModel(best_net)

    # Test snapshot
    MLRAP, TIME_PER_EPOCH = test.test(SNAPSHOT)

    result = np.array([[MLRAP]], dtype='float32')
        
    return result

if __name__ == '__main__':

    cfg.LOG_MODE = 'all'
    r = evaluate()
    log.export()
