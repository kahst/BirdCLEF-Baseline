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
    for spec in audio.specsFromFile(path,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    spec_type=cfg.SPEC_TYPE,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0])):

        # Determine signal to noise ratio
        s2n = audio.signal2noise(spec)
        specs.append(spec)
        noise.append(s2n)

    # Shuffle arrays (we want to select randomly later)
    specs, noise = shuffle(specs, noise, random_state=RANDOM)

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
                print(i + 1, '/', len(afiles), c, afiles[i],)

                # Get specs and signal to noise ratios
                specs, noise = getSpecs(os.path.join(cfg.TRAINSET_PATH, 'train', c, afiles[i]))

                # Save specs if it contains signal
                for s in range(len(specs)):

                    # NaN?
                    if np.isnan(noise[s]):
                        noise[s] = 0.0

                    # Above SIGNAL_THRESHOLD?
                    if noise[s] >= cfg.SPEC_SIGNAL_THRESHOLD:

                        # Create target path for accepted specs
                        filepath = os.path.join(cfg.DATASET_PATH, c)
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)

                        # Count specs
                        spec_cnt += 1

                    else:

                        # Create target path for rejected specs -
                        # but we don't want to save every sample (only 10%)
                        if RANDOM.choice([True, False], p=[0.1, 0.90]):
                            filepath = os.path.join(cfg.NOISE_PATH)
                            if not os.path.exists(filepath):
                                os.makedirs(filepath)
                        else:
                            filepath = None
                    
                    if filepath:
                        # Filename contains s2n-ratio
                        filename = str(int(noise[s] * 10000)).zfill(4) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)

                        # Write to HDD
                        cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)                        

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
