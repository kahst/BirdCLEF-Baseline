# -*- coding: utf-8 -*-
# This file includes functionality for (multi-threaded) batch generation.
# Author: Stefan Kahl, 2018, University of Technology Chemnitz

import sys
sys.path.append("..")

import numpy as np

import config as cfg
from utils import image

RANDOM = cfg.getRandomState()

#################### IMAGE HANDLING #####################
def loadImageAndTarget(sample, augmentation):

    # Load image
    img = image.openImage(sample[0], cfg.IM_DIM)

    # Resize Image
    img = image.resize(img, cfg.IM_SIZE[0], cfg.IM_SIZE[1], mode=cfg.RESIZE_MODE)

    # Do image Augmentation
    if augmentation:
        img = image.augment(img, cfg.IM_AUGMENTATION, cfg.AUGMENTATION_COUNT, cfg.AUGMENTATION_PROBABILITY)

    # Prepare image for net input
    img = image.normalize(img, cfg.ZERO_CENTERED_NORMALIZATION)
    img = image.prepare(img)

    # Get target
    label = sample[1]
    index = cfg.CLASSES.index(label)
    target = np.zeros((len(cfg.CLASSES)), dtype='float32')
    target[index] = 1.0

    return img, target    

#################### BATCH HANDLING #####################
def getDatasetChunk(split):

    #get batch-sized chunks of image paths
    for i in xrange(0, len(split), cfg.BATCH_SIZE):
        yield split[i:i+cfg.BATCH_SIZE]

def getNextImageBatch(split, augmentation=True): 

    #fill batch
    for chunk in getDatasetChunk(split):

        #allocate numpy arrays for image data and targets
        x_b = np.zeros((cfg.BATCH_SIZE, cfg.IM_DIM, cfg.IM_SIZE[1], cfg.IM_SIZE[0]), dtype='float32')
        y_b = np.zeros((cfg.BATCH_SIZE, len(cfg.CLASSES)), dtype='float32')
        
        ib = 0
        for sample in chunk:

            try:
            
                #load image data and class label from path
                x, y = loadImageAndTarget(sample, augmentation)

                #pack into batch array
                x_b[ib] = x
                y_b[ib] = y
                ib += 1

            except:
                continue

        #trim to actual size
        x_b = x_b[:ib]
        y_b = y_b[:ib]

        #instead of return, we use yield
        yield x_b, y_b

#Loading images with CPU background threads during GPU forward passes saves a lot of time
#Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)
def threadedGenerator(generator, num_cached=32):
    
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    #define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    #start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    #run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        try:
            queue.task_done()
            item = queue.get()
        except:
            break

def nextBatch(split, augmentation=True, threaded=True):
    if threaded:
        for x, y in threadedGenerator(getNextImageBatch(split, augmentation)):
            yield x, y
    else:
        for x, y in getNextImageBatch(split, augmentation):
            yield x, y
