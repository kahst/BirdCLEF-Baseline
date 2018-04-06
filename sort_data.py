# Use this script to sort the BirdCLEF 2018 dataset
# Unpack the archive file containing xml-files and wav-files first
# Use config.py to specify source and target paths
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import json

from sklearn.utils import shuffle
from shutil import copyfile
import xmltodict as x2d

import config as cfg

################### METADATA HANDLING ####################
def parseDataset():

    metadata = {}

    # List of wav-files
    wav_path = os.path.join(cfg.TRAINSET_PATH, 'wav')
    wav_files = [f for f in sorted(os.listdir(wav_path))]
    print 'DATASET CONTAINS', len(wav_files), 'WAV_FILES'

    # List all xml-files
    xml_path = os.path.join(cfg.TRAINSET_PATH, 'xml')
    xml_files = [os.path.join(xml_path, f) for f in sorted(os.listdir(xml_path))]
    print 'PARSING', len(xml_files), 'XML-FILES...'

    # Open xml-files and extract metadata
    for i in range(len(xml_files)):

        # Read contnet
        xml = open(xml_files[i], 'r').read()
        data = x2d.parse(xml)

        # The 2017 dataset has no annotated background species
        # We have to handle those separately
        try:
            background = data['Audio']['BackgroundSpecies'].split(',')
        except:
            background = []

        # Create new metadata
        mdata = {'sci-name': data['Audio']['Genus'] + ' ' + data['Audio']['Species'],
                 'species': data['Audio']['VernacularNames'].split(',')[0],
                 'background': background,
                 'filename': data['Audio']['FileName'],
                 'classid': data['Audio']['ClassId']}

        # Save metadata to dict if wav-file exists
        if mdata['filename'] in wav_files:
            if not mdata['classid'] in metadata:
                metadata[mdata['classid']] = []
            metadata[mdata['classid']].append(mdata)

        # Status (parsing the files might take a while)
        if not i % 100:
            print '\t', i, '/', len(xml_files)
                
    print '...DONE!', len(metadata), 'CLASSES IN DATASET'

    return metadata

####################  CREATE SPLITS  #####################
def sortDataset(mdata):

    print 'PARSING CLASSES...'

    # Parse classes
    for c in mdata:

        print '\t', c

        # Determine size of val split (10% but at least 1 file)
        val = max(1, len(mdata[c]) * 0.1)

        # Shuffle list of files
        mdata[c] = shuffle(mdata[c], random_state=cfg.getRandomState())

        # Parse list of files and copy to destination
        for f in mdata[c]:

            # Get class name (we use the sci-name which makes it easier to evaluate with background species)
            # The submission format uses class id only - so we have to figure that out later
            cname = f['sci-name']

            # Make folders
            m_path = os.path.join(cfg.TRAINSET_PATH, 'metadata')
            if not os.path.exists(m_path):
                os.makedirs(m_path)

            t_path = os.path.join(cfg.TRAINSET_PATH, 'train', cname)
            if not os.path.exists(t_path):
                os.makedirs(t_path)

            v_path = os.path.join(cfg.TRAINSET_PATH, 'val', cname)
            if not os.path.exists(v_path):
                os.makedirs(v_path)

            # Copy files
            with open(os.path.join(m_path, f['filename'].rsplit('.')[0] + '.json'), 'w') as mfile:
                json.dump(f, mfile)

            if mdata[c].index(f) < val:
                copyfile(os.path.join(cfg.TRAINSET_PATH, 'wav', f['filename']), os.path.join(v_path, f['filename']))
            else:
                copyfile(os.path.join(cfg.TRAINSET_PATH, 'wav', f['filename']), os.path.join(t_path, f['filename']))

    print '...DONE!'


if __name__ == '__main__':

    # Create metadata for entire dataset
    metadata = parseDataset()

    # Split into train and val, copy files
    sortDataset(metadata)
