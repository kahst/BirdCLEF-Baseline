# This file contains all submission functionality for
# the soundscape test data.
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

def parseTestSet():

    # Status
    log.i('PARSING TEST SET...', new_line=False)
    t = []

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

def getTimestamp(start, end):

    m_s, s_s = divmod(start, 60)
    h_s, m_s = divmod(m_s, 60)
    start = str(h_s).zfill(2) + ":" + str(m_s).zfill(2) + ":" + str(s_s).zfill(2)

    m_e, s_e = divmod(end, 60)
    h_e, m_e = divmod(m_e, 60)
    end = str(h_e).zfill(2) + ":" + str(m_e).zfill(2) + ":" + str(s_e).zfill(2)

    return start + '-' + end

def getClassId(c):

    if c in LABELS:
        return CODES[LABELS.index(c)]
    else:
        print 'MISSING CLASS:', c
        return False

def getSpecBatches(split):

    # Random Seed
    random = cfg.getRandomState()

    # Make predictions for every testfile
    for t in split:

        # Spec batch
        spec_batch = []

        # Keep track of timestamps
        pred_start = 0

        # Get specs for file
        for spec in audio.specsFromFile(t[0],
                                        cfg.SAMPLE_RATE,
                                        cfg.SPEC_LENGTH,
                                        cfg.SPEC_OVERLAP,
                                        cfg.SPEC_MINLEN,
                                        shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                                        fmin=cfg.SPEC_FMIN,
                                        fmax=cfg.SPEC_FMAX):

            # Resize spec
            spec = image.resize(spec, cfg.IM_SIZE[0], cfg.IM_SIZE[1], mode=cfg.RESIZE_MODE)

            # Normalize spec
            spec = image.normalize(spec, cfg.ZERO_CENTERED_NORMALIZATION)

            # Prepare as input
            spec = image.prepare(spec)

            # Add to batch
            if len(spec_batch) > 0:
                spec_batch = np.vstack((spec_batch, spec))
            else:
                spec_batch = spec

            # Batch too large?
            if spec_batch.shape[0] >= cfg.MAX_SPECS_PER_FILE:
                break

            # Do we have enough specs for a prediction?
            if len(spec_batch) >= cfg.SPECS_PER_PREDICTION:

                # Calculate next timestamp
                pred_end = pred_start + cfg.SPEC_LENGTH + ((len(spec_batch) - 1) * (cfg.SPEC_LENGTH - cfg.SPEC_OVERLAP))
                
                # Store prediction
                ts = getTimestamp(int(pred_start), int(pred_end))

                # Advance to next timestamp
                pred_start = pred_end - cfg.SPEC_OVERLAP

                yield spec_batch, t[1], ts, t[0].split(os.sep)[-1]

                # Spec batch
                spec_batch = []
            

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
    for spec_batch, mediaid, timestamp, filename in bg.threadedGenerator(getSpecBatches(TEST)):

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
                    submission += mediaid + ';' + timestamp + ';' + getClassId(p_sorted[i][0]) + ';' + str(p_sorted[i][1]) + '\n'

            # Show sample stats            
            log.i((filename, timestamp), new_line=False)
            log.i(('TOP PREDICTION:', p_sorted[0][0], int(p_sorted[0][1] * 1000) / 10.0, '%'), new_line=True)

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
    with open(cfg.RUN_NAME + '_SOUNDSCAPE_SUBMISSION.txt', 'w') as sfile:
        sfile.write(submission)    
    log.i('DONE!')
