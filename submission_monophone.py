# This file contains all submission functionality for
# the monophonic test data.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import json
import operator

import numpy as np

import config as cfg
import test
from model import lasagne_net as birdnet
from model import lasagne_io as io
from utils import audio
from utils import image
from utils import batch_generator as bg
from utils import log 

################### DATASAT HANDLING ####################
def parseTestSet():

    # Status
    log.i('PARSING TEST SET...', new_line=False)
    t = []

    # Get all sound files
    wav_files = [os.path.join(cfg.TESTSET_PATH, f) for f in sorted(os.listdir(cfg.TESTSET_PATH)) if os.path.splitext(f)[1] in ['.wav']]

    # Parse files
    for f in wav_files:
        t.append((f, os.path.splitext(f)[0].split('_RN')[-1]))

    # Load class ids
    codes = []
    with open('metadata/labelset.txt', 'r') as lfile:
        for line in lfile.readlines():
            codes.append(line.replace('\r\n', '').replace('\n', ''))
    labels = []
    with open('metadata/labelset_latin.txt', 'r') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\r\n', '').replace('\n', ''))

    # Status
    log.i(('Done!', len(t), 'TEST FILES'))

    return t, codes, labels

def getClassId(c):

    if c in LABELS:
        return CODES[LABELS.index(c)]
    else:
        print 'MISSING CLASS:', c
        return None       
    
def runTest(SNAPSHOTS, TEST):

    # Do we have more than one snapshot?
    if not isinstance(SNAPSHOTS, (list, tuple)):
        SNAPSHOTS = [SNAPSHOTS]
        
    # Load snapshots
    test_functions = []
    for s in SNAPSHOTS:

        # Settings
        NET = s['net']
        cfg.CLASSES = s['classes']
        cfg.IM_DIM = s['im_dim']
        cfg.IM_SIZE = s['im_size']        

        # Compile test function
        test_net = birdnet.test_function(NET, hasTargets=False, layer_index=-1)
        test_functions.append(test_net)       
    
    # Status
    log.i('START TESTING...')

    # Make predictions
    submission = ''
    cnt = 1
    for spec_batch, mediaid, filename in bg.threadedGenerator(test.getSpecBatches(TEST)):

        try:

            # Prediction
            prediction_batch = []
            for test_func in test_functions:
                if len(prediction_batch) == 0:
                    prediction_batch = test_func(spec_batch)
                else:
                    prediction_batch += test_func(spec_batch)
            prediction_batch /= len(test_functions)

            # Eliminate the scores for 'Noise'
            if 'Noise' in cfg.CLASSES:
                prediction_batch[: , cfg.CLASSES.index('Noise')] = np.min(prediction_batch)
            
            # Prediction pooling
            p_pool = test.predictionPooling(prediction_batch)

            # Get class labels
            p_labels = {}
            for i in range(p_pool.shape[0]):
                p_labels[cfg.CLASSES[i]] = p_pool[i]

            # Sort by score
            p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

            # Add scores to submission
            for i in range(min(100, len(p_sorted))):
                if getClassId(p_sorted[i][0]):
                    submission += mediaid + ';' + getClassId(p_sorted[i][0]) + ';' + str(p_sorted[i][1]) + ';' + str(i + 1) + '\n'

            # Show sample stats            
            log.i((cnt, filename), new_line=False)
            log.i(('TOP PREDICTION:', p_sorted[0][0], int(p_sorted[0][1] * 1000) / 10.0, '%'), new_line=True)
            cnt += 1

        except KeyboardInterrupt:
            cfg.DO_BREAK = True
            break

    # Status
    log.i('DONE TESTING!')

    return submission

if __name__ == '__main__':

    # Parse Testset
    TEST, CODES, LABELS = parseTestSet()

    # Load trained models
    if not isinstance(cfg.TEST_MODELS, (list, tuple)):
        cfg.TEST_MODELS = [cfg.TEST_MODELS]
    SNAPSHOTS = []
    for test_model in cfg.TEST_MODELS:
        SNAPSHOTS.append(io.loadModel(test_model))

    # Generate submission
    submission = runTest(SNAPSHOTS, TEST)

    # Write submission to file
    log.i('WRITING SUBMISSION...', new_line=False)
    with open(cfg.RUN_NAME + '_MONOPHONE_SUBMISSION.txt', 'w') as sfile:
        sfile.write(submission)    
    log.i('DONE!')
