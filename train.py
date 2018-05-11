# This file contains all the training functionality including
# dataset parsing and snapshot export
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import operator
import time

import numpy as np
from sklearn.utils import shuffle

import config as cfg
from model import lasagne_net as birdnet
from model import learning_rate as lr
from model import lasagne_io as io
from utils import image
from utils import batch_generator as bg
from utils import stats
from utils import metrics
from utils import log

################### DATASAT HANDLING ####################
def isValidClass(c, path):

    # Do we have noise samples we want to include?
    if c in ['Noise']:
        return True

    # Class in S2N interval?
    if (int(path.split('_')[0]) >= cfg.S2N_INTERVAL[0] and int(path.split('_')[0]) <= cfg.S2N_INTERVAL[1]):
        return True
    else:
        return False

def parseDataset():

    # Random Seed
    random = cfg.getRandomState()

    # We use subfolders as class labels
    classes = [folder for folder in sorted(os.listdir(cfg.DATASET_PATH)) if folder in cfg.CLASS_WHITELIST or len(cfg.CLASS_WHITELIST) == 0]
    if not cfg.SORT_CLASSES_ALPHABETICALLY:
        classes = shuffle(classes, random_state=random)
    classes = classes[:cfg.MAX_CLASSES]

    # Now we enlist all image paths for each class
    images = []
    tclasses = []
    sample_count = {}
    for c in classes:
        c_images = [os.path.join(cfg.DATASET_PATH, c, path) for path in shuffle(os.listdir(os.path.join(cfg.DATASET_PATH, c)), random_state=random) if isValidClass(c, path)][:cfg.MAX_SAMPLES_PER_CLASS]
        
        sample_count[c] = len(c_images)
        images += c_images
        
        # Do we want to correct class imbalance?
        # This will affect validation scores as we use some samples in TRAIN and VAL
        while sample_count[c] < cfg.MIN_SAMPLES_PER_CLASS:
            images += [c_images[random.randint(0, len(c_images))]]
            sample_count[c] += 1

    # Add labels to image paths
    for i in range(len(images)):
        path = images[i]
        label = images[i].split(os.sep)[-2]
        images[i] = (path, label)

    # Shuffle image paths
    images = shuffle(images, random_state=random)

    # Validation split
    vsplit = int(len(images) * cfg.VAL_SPLIT)
    train = images[:-vsplit]
    val = images[-vsplit:]

    # Show some stats
    log.i(("CLASSES:", len(classes)))
    log.i(( "CLASS LABELS:", sorted(sample_count.items(), key=operator.itemgetter(1))))
    log.i(("TRAINING IMAGES:", len(train)))
    log.i(("VALIDATION IMAGES:", len(val)))

    return classes, train, val

####################### TRAINING ########################
def train(NET, TRAIN, VAL):

    # Random Seed
    random = cfg.getRandomState()
    image.resetRandomState()

    # Load pretrained model
    if cfg.PRETRAINED_MODEL_NAME:
        snapshot = io.loadModel(cfg.PRETRAINED_MODEL_NAME)
        NET = io.loadParams(NET, snapshot['params'])

    # Load teacher models
    teach_funcs = []
    if len(cfg.TEACHER) > 0:
        for t in cfg.TEACHER:
            snapshot = io.loadModel(t)
            TEACHER = snapshot['net']
            teach_funcs.append(birdnet.test_function(TEACHER, hasTargets=False))

    # Compile Theano functions
    train_net = birdnet.train_function(NET)
    test_net = birdnet.test_function(NET)

    # Status
    log.i("START TRAINING...")

    # Train for some epochs...
    for epoch in range(cfg.EPOCH_START, cfg.EPOCHS + 1):

        try:

            # Stop?
            if cfg.DO_BREAK:
                break

            # Clear stats for every epoch
            stats.clearStats()
            stats.setValue('sample_count', len(TRAIN) + len(VAL))

            # Start timer
            stats.tic('epoch_time')
            
            # Shuffle dataset (this way we get "new" batches every epoch)
            TRAIN = shuffle(TRAIN, random_state=random)

            # Iterate over TRAIN batches of images
            for image_batch, target_batch in bg.nextBatch(TRAIN):

                # Show progress
                stats.showProgress(epoch)

                # If we have a teacher, we use that model to get new targets
                if len(teach_funcs) > 0:
                    target_batch = np.zeros((len(teach_funcs), target_batch.shape[0], target_batch.shape[1]), dtype='float32')
                    for i in range(len(teach_funcs)):
                        target_batch[i] = teach_funcs[i](image_batch)
                    target_batch = np.mean(target_batch, axis=0)
                
                # Calling the training functions returns the current loss
                loss = train_net(image_batch, target_batch, lr.dynamicLearningRate(cfg.LR_SCHEDULE, epoch))
                stats.setValue('train loss', loss, 'append')
                stats.setValue('batch_count', 1, 'add')

                # Stop?
                if cfg.DO_BREAK:
                    break

            # Iterate over VAL batches of images
            for image_batch, target_batch in bg.nextBatch(VAL, False, True):

                # Calling the test function returns the net output, loss and accuracy
                prediction_batch, loss, acc = test_net(image_batch, target_batch)
                stats.setValue('val loss', loss, 'append')
                stats.setValue('val acc', acc, 'append')
                stats.setValue('batch_count', 1, 'add')
                stats.setValue('lrap', [metrics.lrap(prediction_batch, target_batch)], 'add')

                # Show progress
                stats.showProgress(epoch)

                # Stop?
                if cfg.DO_BREAK:
                    break

            # Show stats for epoch
            stats.showProgress(epoch, done=True)
            stats.toc('epoch_time')
            log.r(('TRAIN LOSS:', np.mean(stats.getValue('train loss'))), new_line=False)
            log.r(('VAL LOSS:', np.mean(stats.getValue('val loss'))), new_line=False)
            log.r(('VAL ACC:', int(np.mean(stats.getValue('val acc')) * 10000) / 100.0, '%'), new_line=False)          
            log.r(('MLRAP:', int(np.mean(stats.getValue('lrap')) * 1000) / 1000.0), new_line=False)
            log.r(('TIME:', stats.getValue('epoch_time'), 's'))

            # Save snapshot?
            if not epoch % cfg.SNAPSHOT_EPOCHS:
                io.saveModel(NET, cfg.CLASSES, epoch)
                io.saveParams(NET, cfg.CLASSES, epoch)

            # New best net?
            if np.mean(stats.getValue('lrap')) > stats.getValue('best_mlrap'):
                stats.setValue('best_net', NET, static=True)
                stats.setValue('best_epoch', epoch, static=True)
                stats.setValue('best_mlrap', np.mean(stats.getValue('lrap')), static=True)

            # Early stopping?
            if epoch - stats.getValue('best_epoch') >= cfg.EARLY_STOPPING_WAIT:
                log.i('EARLY STOPPING!')
                break

            # Stop?
            if cfg.DO_BREAK:
                break

        except KeyboardInterrupt:
            log.i('KeyboardInterrupt')
            cfg.DO_BREAK = True
            break

    # Status
    log.i('TRAINING DONE!')
    log.r(('BEST MLRAP:', stats.getValue('best_mlrap'), 'EPOCH:', stats.getValue('best_epoch')))

    # Save best model and return
    io.saveParams(stats.getValue('best_net'), cfg.CLASSES, stats.getValue('best_epoch'))
    return io.saveModel(stats.getValue('best_net'), cfg.CLASSES, stats.getValue('best_epoch'))

if __name__ == '__main__':

    cfg.CLASSES, TRAIN, VAL = parseDataset()
    NET = birdnet.build_model()

    net_name = train(NET, TRAIN, VAL)
    
