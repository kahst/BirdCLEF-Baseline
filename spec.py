# Use this file to extract spectrograms from your training set
# which needs to be organized in subfolders representing class names.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import time

import numpy as np
import cv2

from sklearn.utils import shuffle

import config as cfg
from utils import audio
from utils import log

######################## CONFIG #########################
RANDOM = cfg.getRandomState()

######################### SPEC ##########################
def getSpecs(path):
    
    specs = []
    noise = []

    # Get mel-specs for file
    for mspec in audio.specsFromFile(path,
                                     rate=cfg.SAMPLE_RATE,
                                     seconds=cfg.SPEC_LENGTH,
                                     overlap=cfg.SPEC_OVERLAP,
                                     minlen=cfg.SPEC_MINLEN,
                                     fmin=cfg.SPEC_FMIN,
                                     fmax=cfg.SPEC_FMAX,
                                     shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0])):

        # Determine signal to noise ratio
        s2n = audio.signal2noise(mspec)
        specs.append(mspec)
        noise.append(s2n)

    # Shuffle arrays (we want to select randomly later)
    specs = shuffle(specs, random_state=RANDOM)
    noise = shuffle(noise, random_state=RANDOM)

    return specs, noise

def parseDataset():

    # List of classes, subfolders as class names
    CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train')))]

    # Parse every class
    for c in CLASSES:

        # List all audio files
        afiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train', c)))]

        # Calculate maximum specs per file
        max_specs = cfg.MAX_SPECS_PER_CLASS // len(afiles) + 1

        # Get specs for every audio file
        for i in range(len(afiles)):

            spec_cnt = 0

            try:

                # Stats
                print i + 1, '/', len(afiles), c, afiles[i],

                # Get specs and signal to noise ratios
                specs, noise = getSpecs(os.path.join(cfg.TRAINSET_PATH, 'train', c, afiles[i]))

                # Save specs if it contains signal
                for s in range(len(specs)):

                    # NaN?
                    if np.isnan(noise[s]):
                        noise[s] = 0.0

                    # Above SIGNAL_THRESHOLD?
                    if noise[s] >= cfg.SPEC_SIGNAL_THRESHOLD:

                        # Create target path
                        filepath = os.path.join(cfg.DATASET_PATH, c)
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)

                        # Filename contains s2n-ratio
                        filename = str(int(noise[s] * 1000)).zfill(3) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)

                        # Write to HDD
                        cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)

                        # Count specs
                        spec_cnt += 1

                        # Do we have enough specs already?
                        if spec_cnt >= max_specs:
                            break

                # Stats
                log.i((spec_cnt, 'specs'))       

            except:
                log.e((spec_cnt, 'specs', 'ERROR DURING SPEC EXTRACT'))
                continue



if __name__ == '__main__':

    parseDataset()
