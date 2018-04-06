# This file contains model I/O functionality.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import os
import pickle

from lasagne import layers as l

import config as cfg
from utils import log

sys.setrecursionlimit(10000)

def saveModel(net, classes, epoch):
    log.i("EXPORTING MODEL...", new_line=False)
    net_filename = cfg.MODEL_PATH + cfg.RUN_NAME + "_model_epoch_" + str(epoch) + ".pkl"
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)
    with open(net_filename, 'w') as f:
        
        #We want to save the model architecture with all params and trained classes
        data = {'net': net, 'classes':classes, 'run_name': cfg.RUN_NAME, 'epoch':epoch, 'im_size':cfg.IM_SIZE, 'im_dim':cfg.IM_DIM}        
        pickle.dump(data, f)

    log.i("DONE!")

    return os.path.split(net_filename)[-1]

def loadModel(filename):
    log.i("IMPORTING MODEL...", new_line=False)
    net_filename = cfg.MODEL_PATH + filename

    with open(net_filename, 'rb') as f:
        model = pickle.load(f)

    log.i("DONE!")
    
    return model

def saveParams(net, classes, epoch):

    log.i("EXPORTING MODEL PARAMS...", new_line=False)
    net_filename = cfg.MODEL_PATH + cfg.RUN_NAME + "_model_params_epoch_" + str(epoch) + ".pkl"
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)
    with open(net_filename, 'w') as f:
        
        #We want to save the model params only and trained classes
        params = l.get_all_param_values(net)
        data = {'params': params, 'classes':classes, 'run_name': cfg.RUN_NAME, 'epoch':epoch, 'im_size':cfg.IM_SIZE, 'im_dim':cfg.IM_DIM}        
        pickle.dump(data, f)

    log.i("DONE!")

    return os.path.split(net_filename)[-1]    

def loadParams(net, params):

    log.i("IMPORTING MODEL PARAMS...", new_line=False)
    if cfg.LOAD_OUTPUT_LAYER:
        l.set_all_param_values(net, params)
    else:
        l.set_all_param_values(l.get_all_layers(net)[:-1], params[:-2])

    log.i("DONE!")
    
    return net
