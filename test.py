# This file contains all testing functionality including
# dataset parsing and evaluation.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import json
import operator

import numpy as np
from sklearn.utils import shuffle

import config as cfg
from model import lasagne_net as birdnet
from model import lasagne_io as io
from utils import audio
from utils import image
from utils import batch_generator as bg
from utils import metrics
from utils import stats
from utils import log 

################### DATASAT HANDLING ####################
def parseTestSet():

    # Random Seed
    random = cfg.getRandomState()

    # Status
    log.i('PARSING TEST SET...', new_line=False)
    TEST = []

    # List of test files
    fnames = []
    for path, dirs, files in os.walk(cfg.TESTSET_PATH):
        if path.split(os.sep)[-1] in cfg.CLASSES:
            scnt = 0
            for f in files:
                fnames.append(os.path.join(path, f))
                scnt += 1
                if scnt >= cfg.MAX_TEST_SAMPLES_PER_CLASS and cfg.MAX_TEST_SAMPLES_PER_CLASS > 0:
                    break
    fnames = sorted(shuffle(fnames, random_state=random)[:cfg.MAX_TEST_FILES])

    # Get ground truth from metadata
    for f in fnames:

        # Metadata path
        m_path = os.path.join(cfg.METADATA_PATH, f.split(os.sep)[-1].split('.')[0] + '.json')

        # Load JSON
        with open(m_path) as jfile:
            data = json.load(jfile)

        # Get Species (+ background species)
        # Only species present in the trained classes are relevant for the metric
        # Still, we are adding anything we have right now and sort it out later
        if cfg.TEST_WITH_BG_SPECIES:
            bg = data['background']
        else:
            bg = []
        species = [data['sci-name']] + bg

        # Add data to test set
        TEST.append((f, species))

    # Status
    log.i('DONE!')
    log.i(('TEST FILES:', len(TEST)))

    return TEST

####################### TESTING #########################
labels = []
mdata = {}
def applyMetadata(fname, p):

    global labels
    global mdata

    if len(labels) == 0:
        with open('labelset_latin', 'r') as lfile:
            for line in lfile.readlines():
                labels.append(line.replace('\r\n', '').replace('\n', ''))

    if len(mdata) == 0:
        with open('meta_prob.tsv', 'r') as mfile:
            for line in mfile.readlines():
                d = line.replace('\r\n', '').replace('\n', '').split('\t')
                mdata[d[0].split(os.sep)[-1]] = d[1:]

    probs = (np.array(mdata[fname], dtype='float32') + 1)
    probs[probs >= 1.0] = 1.5
    for i in range(len(p)):
        if cfg.CLASSES[i] in labels:
            p[i] *= probs[labels.index(cfg.CLASSES[i])]

    return p


def predictionPooling(p):
    
    #You can test different prediction pooling strategies here
    if p.ndim == 2:

        try:

            # Median filtered pooling for monophonic recordings
            row_median = np.median(p, axis=1, keepdims=True)
            p[p < row_median * 1.5] = 0.0
            p_pool = np.mean((p * 2) ** 2, axis=0)
            p_pool -= p_pool.min()
            if p_pool.max() > 1.0:
                p_pool /= p_pool.max()

            # Mean exponential pooling for monophonic recordings
            #p_pool = np.mean((p * 2) ** 2, axis=0)
            #p_pool[p_pool > 1.0] = 1.0

            # Simple average pooling
            #p_pool = np.mean(p, axis=0)
            #p_pool = sigmoid(p_pool)

        except:
            p_pool = cfg.getRandomState().normal(0.0, 1.0, (p.shape[1]))
        
    else:
        p_pool = p

    return p_pool

def getSpecBatches(split):

    # Random Seed
    random = cfg.getRandomState()

    # Make predictions for every testfile
    for t in split:

        # Spec batch
        spec_batch = []

        # Get specs for file
        for spec in audio.specsFromFile(t[0],
                                        cfg.SAMPLE_RATE,
                                        cfg.SPEC_LENGTH,
                                        cfg.SPEC_OVERLAP,
                                        cfg.SPEC_MINLEN,
                                        shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                                        fmin=cfg.SPEC_FMIN,
                                        fmax=cfg.SPEC_FMAX,
                                        spec_type=cfg.SPEC_TYPE):

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

        # No specs?
        if len(spec_batch) == 0:
            spec = random.normal(0.0, 1.0, (cfg.IM_SIZE[1], cfg.IM_SIZE[0]))
            spec_batch = image.prepare(spec)

        # Shuffle spec batch
        spec_batch = shuffle(spec_batch, random_state=random)

        # yield batch, labels and filename
        yield spec_batch[:cfg.MAX_SPECS_PER_FILE], t[1], t[0].split(os.sep)[-1]

def test(SNAPSHOTS):

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

    # Parse Testset
    TEST = parseTestSet()
    
    # Status
    log.i('START TESTING...')
    stats.clearStats()    
    stats.tic('test_time')

    # Make predictions
    for spec_batch, labels, filename in bg.threadedGenerator(getSpecBatches(TEST)):

        try:

            # Status
            stats.tic('pred_time')

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
            p_pool = predictionPooling(prediction_batch)

            # Get class labels
            p_labels = {}
            for i in range(p_pool.shape[0]):
                p_labels[cfg.CLASSES[i]] = p_pool[i]

            # Sort by score
            p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)  

            # Calculate MLRAP (MRR for single labels)
            targets = np.zeros(p_pool.shape[0], dtype='float32')
            for label in labels:
                if label in cfg.CLASSES:
                    targets[cfg.CLASSES.index(label)] = 1.0                     
            lrap = metrics.lrap(np.expand_dims(p_pool, 0), np.expand_dims(targets, 0))
            stats.setValue('lrap', lrap, mode='append')            

            # Show sample stats            
            log.i((filename), new_line=True)
            log.i(('\tLABELS:', labels), new_line=True)
            log.i(('\tTOP PREDICTION:', p_sorted[0][0], int(p_sorted[0][1] * 1000) / 10.0, '%'), new_line=True)
            log.i(('\tLRAP:', int(lrap * 1000) / 1000.0), new_line=False)
            log.i(('\tMLRAP:', int(np.mean(stats.getValue('lrap')) * 1000) / 1000.0), new_line=True)
            
            # Save some stats
            if p_sorted[0][0] == labels[0]:
                stats.setValue('top1_correct', 1, 'add')
                stats.setValue('top1_confidence', p_sorted[0][1], 'append')
            else:
                stats.setValue('top1_incorrect', 1, 'add')
            stats.toc('pred_time')
            stats.setValue('time_per_batch', stats.getValue('pred_time'), 'append')

        except KeyboardInterrupt:
            cfg.DO_BREAK = True
            break
        except:
            log.e('ERROR WHILE TRAINING!')
            continue

    # Stats
    stats.toc('test_time')
    log.i(('TESTING DONE!', 'TIME:', stats.getValue('test_time'), 's'))
    log.r(('FINAL MLRAP:', np.mean(stats.getValue('lrap'))))
    log.r(('TOP 1 ACCURACY:', max(0, float(stats.getValue('top1_correct')) / (stats.getValue('top1_correct') + stats.getValue('top1_incorrect')))))
    log.r(('TOP 1 MEAN CONFIDENCE:',max(0, np.mean(stats.getValue('top1_confidence')))))
    log.r(('TIME PER BATCH:', int(np.mean(stats.getValue('time_per_batch')) * 1000), 'ms'))
        
    return np.mean(stats.getValue('lrap')), int(np.mean(stats.getValue('time_per_file')) * 1000)

if __name__ == '__main__':
    
    # Load trained models
    if not isinstance(cfg.TEST_MODELS, (list, tuple)):
        cfg.TEST_MODELS = [cfg.TEST_MODELS]
    SNAPSHOTS = []
    for test_model in cfg.TEST_MODELS:
        SNAPSHOTS.append(io.loadModel(test_model))    

    # Test snapshots
    MLRAP, TIME = test(SNAPSHOTS)
    
    
