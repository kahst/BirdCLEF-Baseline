# This file contains the model architecture as well as
# Theano specific function compilation.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import time
import math

import numpy as np

import theano
import theano.tensor as T

from lasagne import layers as l
from lasagne import nonlinearities as nl
from lasagne import init
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

try:
    from lasagne.layers.dnn import batch_norm_dnn as l_batch_norm
except ImportError:
    from lasagne.layers import batch_norm as l_batch_norm 

import config as cfg
import lasagne_io as io
from utils import log

from lasagne import random as lasagne_random

################ ADDITIONAL FUNCTUALITY #################
def batch_norm(layer):
    if cfg.BATCH_NORM:
        return l_batch_norm(layer)
    else:
        return layer

def nonlinearity(name):

    nonlinearities = {'rectify': nl.rectify,
                     'relu': nl.rectify,
                     'lrelu': nl.LeakyRectify(0.01),
                     'vlrelu': nl.LeakyRectify(0.33),
                     'elu': nl.elu,
                     'softmax': nl.softmax,
                     'sigmoid': nl.sigmoid,
                     'identity':nl.identity}

    return nonlinearities[name]

def initialization(name):

    initializations = {'sigmoid':init.HeNormal(gain=1.0),
            'softmax':init.HeNormal(gain=1.0),
            'elu':init.HeNormal(gain=1.0),
            'relu':init.HeNormal(gain=math.sqrt(2)),
            'lrelu':init.HeNormal(gain=math.sqrt(2/(1+0.01**2))),
            'vlrelu':init.HeNormal(gain=math.sqrt(2/(1+0.33**2))),
            'rectify':init.HeNormal(gain=math.sqrt(2)),
            'identity':init.HeNormal(gain=math.sqrt(2))
            }

    return initializations[name]

################## BUILDING THE MODEL ###################
def build_model():

    log.i('BUILDING MODEL...')

    # Random Seed
    lasagne_random.set_rng(cfg.getRandomState())

    # Input layer for images
    net = l.InputLayer((None, cfg.IM_DIM, cfg.IM_SIZE[1], cfg.IM_SIZE[0]))

    # Stride size (as an alternative to max pooling)
    if cfg.MAX_POOLING:
        s = 1
    else:
        s = 2

    # Convolutinal layer groups
    for i in range(len(cfg.FILTERS)):
        
        # 3x3 Convolution + Stride
        net = batch_norm(l.Conv2DLayer(net,
                                       num_filters=cfg.FILTERS[i],
                                       filter_size=cfg.KERNEL_SIZES[i],
                                       num_groups=cfg.NUM_OF_GROUPS[i],
                                       pad='same',
                                       stride=s,
                                       W=initialization(cfg.NONLINEARITY),
                                       nonlinearity=nonlinearity(cfg.NONLINEARITY)))

        # Pooling layer
        if cfg.MAX_POOLING:
            net = l.MaxPool2DLayer(net, pool_size=2)

        # Dropout Layer (we support different types of dropout)
        if cfg.DROPOUT_TYPE == 'channels':
            net = l.dropout_channels(net, p=cfg.DROPOUT)
        elif cfg.DROPOUT_TYPE == 'locations':
            net = l.dropout_locations(net, p=cfg.DROPOUT)
        else:
            net = l.DropoutLayer(net, p=cfg.DROPOUT)
        
        log.i(('\tGROUP', i + 1, 'OUT SHAPE:', l.get_output_shape(net)))
    
    # Final 1x1 Convolution
    net = batch_norm(l.Conv2DLayer(net,
                                   num_filters=cfg.FILTERS[i] * 2,
                                   filter_size=1,
                                   W=initialization('identity'),
                                   nonlinearity=nonlinearity('identity')))

    log.i(('\tFINAL CONV OUT SHAPE:', l.get_output_shape(net)))
    
    # Global Pooling layer (default mode = average)
    net = l.GlobalPoolLayer(net)
    log.i(("\tFINAL POOLING SHAPE:", l.get_output_shape(net)))

    # Classification Layer (Softmax)
    net = l.DenseLayer(net, len(cfg.CLASSES), nonlinearity=nonlinearity('softmax'), W=initialization('softmax'))
    
    log.i(("\tFINAL NET OUT SHAPE:", l.get_output_shape(net)))
    log.i("...DONE!")

    # Model stats
    log.i(("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS"))
    log.i(("MODEL HAS", l.count_params(net), "PARAMS"))

    return net

######################## I/O ############################
def loadPretrained(net):

    if cfg.MODEL_NAME:

        # Load saved model
        n, c = io.loadModel(cfg.MODEL_NAME)

        # Set params
        params = l.get_all_param_values(n)
        if cfg.LOAD_OUTPUT_LAYER:
            l.set_all_param_values(net, params)
        else:
            l.set_all_param_values(l.get_all_layers(net)[:-1], params[:-2])

    return net

#################### LOSS FUNCTION ######################
def calc_loss(prediction, targets):

    # Categorical crossentropy is the best choice for a multi-class softmax output
    loss = T.mean(objectives.categorical_crossentropy(prediction, targets))
    
    return loss

def loss_function(net, prediction, targets):        

    # We use L2 Norm for regularization
    l2_reg = regularization.regularize_layer_params(net, regularization.l2) * cfg.L2_WEIGHT

    # Calculate the loss
    loss = calc_loss(prediction, targets) + l2_reg

    return loss

################# ACCURACY FUNCTION #####################
def calc_accuracy(prediction, targets):

    # We can use the lasagne objective categorical_accuracy to determine the top1 single label accuracy
    a = T.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
    
    return a

def accuracy_function(net, prediction, targets):

    # Calculate accuracy
    accuracy = calc_accuracy(prediction, targets)

    return accuracy

####################### UPDATES #########################
def net_updates(net, loss, lr):    
                        
    # Get all trainable parameters (weights) of our net
    params = l.get_all_params(net, trainable=True)

    # We use the adam update, other options are available
    if cfg.OPTIMIZER == 'adam':
        param_updates = updates.adam(loss, params, learning_rate=lr, beta1=0.9)
    elif cfg.OPTIMIZER == 'nesterov':
        param_updates = updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    elif cfg.OPTIMIZER == 'sgd':
        param_updates = updates.sgd(loss, params, learning_rate=lr)    

    return param_updates

#################### TRAIN FUNCTION #####################
def train_function(net):

    # We use dynamic learning rates which change after some epochs
    lr_dynamic = T.scalar(name='learning_rate')

    # Theano variable for the class targets
    targets = T.matrix('targets', dtype=theano.config.floatX)

    # Get the network output
    prediction = l.get_output(net)
    
    # The theano train functions takes images and class targets as input
    log.i("COMPILING TRAIN FUNCTION...", new_line=False)
    start = time.time()
    loss = loss_function(net, prediction, targets)
    updates = net_updates(net, loss, lr_dynamic)
    train_net = theano.function([l.get_all_layers(net)[0].input_var, targets, lr_dynamic], loss, updates=updates, allow_input_downcast=True)
    log.i(("DONE! (", int(time.time() - start), "s )"))

    return train_net

################# PREDICTION FUNCTION ####################
def test_function(net, hasTargets=True):    

    # We need the prediction function to calculate the validation accuracy
    # this way we can test the net during/after training
    # We need a version with targets and one without
    prediction = l.get_output(net, deterministic=True)

    log.i("COMPILING TEST FUNCTION...", new_line=False)
    start = time.time()
    if hasTargets:
        # Theano variable for the class targets
        targets = T.matrix('targets', dtype=theano.config.floatX)
        
        loss = loss_function(net, prediction, targets)
        accuracy = accuracy_function(net, prediction, targets)
        
        test_net = theano.function([l.get_all_layers(net)[0].input_var, targets], [prediction, loss, accuracy], allow_input_downcast=True)

    else:
        test_net = theano.function([l.get_all_layers(net)[0].input_var], prediction, allow_input_downcast=True)
        
    log.i(("DONE! (", int(time.time() - start), "s )"))

    return test_net

