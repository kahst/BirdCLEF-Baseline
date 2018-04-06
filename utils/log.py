# This files handles log messages with support for different log-modes.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import config as cfg

log = ''
def show(s, new_line=False):

    global log

    if new_line:
        print s
        log += str(s) + '\n'
    else:
        print s,
        log += str(s)

def i(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'info']:
        if isinstance(s, (list, tuple)):
            for st in s:
                show(st)
        else:
            show(s)

        if new_line:
            show('', True)

def p(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'progress']:
        if isinstance(s, (list, tuple)):
            for st in s:
                show(st)
        else:
            show(s)

        if new_line:
            show('', True)

def e(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'error']:
        if isinstance(s, (list, tuple)):
            for st in s:
                show(st)
        else:
            show(s)

        if new_line:
            show('', True)

def r(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'result']:
        if isinstance(s, (list, tuple)):
            for st in s:
                show(st)
        else:
            show(s)

        if new_line:
            show('', True)

def clear():

    global log
    log = ''

def export():

    with open(cfg.LOG_FILE, 'w') as lfile:
        lfile.write(log)
    
