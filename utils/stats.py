# This file contains basic statistics functionality. All runtime values are stored
# and can be referenced by name for further usage.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import copy
import time

import config as cfg
from utils import log

def clearStats(clear_all=False):

    # Clears all recorded values
    # Exceptions are:

    # Permanent values cannot be deleted
    if not 'permanent' in cfg.STATS:
        cfg.STATS['permanent'] = {}
    p = copy.deepcopy(cfg.STATS['permanent'])

    # Static values will only be deleted if said so
    if not 'static' in cfg.STATS:
        cfg.STATS['static'] = {}
    s = copy.deepcopy(cfg.STATS['static'])

    if not clear_all:        
        cfg.STATS = {'static':s, 'permanent':p}
    else:
        cfg.STATS = {'static':{}, 'permanent':p}

    # Copy values
    for name in cfg.STATS['permanent']:
        cfg.STATS[name] = cfg.STATS['permanent'][name]

    for name in cfg.STATS['static']:
        cfg.STATS[name] = cfg.STATS['static'][name]

def tic(name):

    if not 'times' in cfg.STATS:
        cfg.STATS['times'] = {}

    cfg.STATS['times'][name] = time.time()

def toc(name):

    s = int(abs(time.time() - cfg.STATS['times'][name]) * 100) / 100.0
    setValue(name, s)

def setValue(name, v, mode='replace', static=False, permanent=False):

    if not name in cfg.STATS:
        if mode == 'replace' or mode == 'add':
            cfg.STATS[name] = v
        else:
            cfg.STATS[name] = [v]
    else:
        if mode == 'append':
            cfg.STATS[name].append(v)
        elif mode == 'add':
            cfg.STATS[name] += v
        else:
            cfg.STATS[name] = v

    if static:
        cfg.STATS['static'][name] = cfg.STATS[name]

    if permanent:
        cfg.STATS['permanent'][name] = cfg.STATS[name]

def getValue(name, default=-1):

    if name in cfg.STATS:
        return cfg.STATS[name]
    else:
        return default

last_update = -1
def showProgress(epoch, done=False):

    global last_update

    # First call?
    if not 'batch_count' in cfg.STATS:
        bcnt = 0
    else:
        bcnt = cfg.STATS['batch_count']

    # Calculate number of batches to train
    total_batches = cfg.STATS['sample_count'] // cfg.BATCH_SIZE + 1

    # Current progess
    if not done:
        if bcnt == 0:
            log.p(('EPOCH', epoch, '['), new_line=False)
        else:
            p = bcnt * 100 / total_batches
            if not p % 5 and not p == last_update:
                log.p('=', new_line=False)
                last_update = p
    else:
        log.p(']', new_line=False)

# Clear on first load
clearStats(True)
    

    
