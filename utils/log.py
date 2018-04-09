# This files handles log messages with support for different log-modes.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import sys
sys.path.append("..")

import config as cfg

log = ''
def show(s, new_line=False):

    global log

    if isinstance(s, (list, tuple)):
        for i in range(len(s)):
            print s[i],
            log += str(s[i])
            if i < len(s) - 1:
                log += ' '
    else:
        print s,
        log += str(s)

    if new_line:
        print ''
        log += '\n'
    else:
        log += ' '
    

def i(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'info']:
        show(s, new_line)

def p(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'progress']:        
        show(s, new_line)            

def e(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'error']:        
        show(s, new_line)       

def r(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'result']:        
        show(s, new_line)       

def clear():

    global log
    log = ''

def export():

    with open(cfg.LOG_FILE, 'w') as lfile:
        lfile.write(log)
    
