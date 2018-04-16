# This file stores all settings used for spectrogram extraction, training and testing
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os
import numpy as np

# Fixed random seed
def getRandomState():

    RANDOM_SEED = 1337
    RANDOM = np.random.RandomState(RANDOM_SEED)

    return RANDOM

########################  DATASET  ########################

# Path settings (train audio and xml files, specs, test audio files and json metadata)
# Use 'sort_data.py' to organize the BirdCLEF dataset accordingly
# Extract the BirdCLEF TrainingSet data into TRAINSET_PATH
TRAINSET_PATH = 'datasets/TrainingSet/'
DATASET_PATH = 'datasets/spec/'
NOISE_PATH = 'datasets/noise/'
TESTSET_PATH = 'datasets/val/'
METADATA_PATH = 'datasets/metadata/'

# Maximum number of classes to use (None = no limit)
MAX_CLASSES = None

# If not sorted, using only a subset of classes (MAX_CLASSES) will select classes randomly
SORT_CLASSES_ALPHABETICALLY = False  

# Specify minimum and maximum amount of samples (specs) per class
MIN_SAMPLES_PER_CLASS = -1   # -1 = no minimum                                      
MAX_SAMPLES_PER_CLASS = None # None = no limit

# Specify the signal-to-noise interval you want to pick samples from (filename contains value)
S2N_INTERVAL = [50, 1200]

# Size of validation split (0.05 = 5%)
VAL_SPLIT = 0.05

######################  SPECTROGRAMS  ######################

# Sample rate for recordings, other sampling rates will force re-sampling
SAMPLE_RATE = 44100

# Specify min and max frequency for low and high pass
SPEC_FMIN = 300
SPEC_FMAX = 15000

# Define length of chunks for spec generation, overlap of chunks and chunk min length
SPEC_LENGTH = 1.0
SPEC_OVERLAP = 0.25
SPEC_MINLEN = 1.0

# Threshold for distinction between noise and signal
SPEC_SIGNAL_THRESHOLD = 0.0001

# Limit the amount of specs per class when extracting spectrograms (None = no limit)
MAX_SPECS_PER_CLASS = 1000

#########################  IMAGE  #########################

# Number of channels
IM_DIM = 1

# Image size (width, height), should be the same as spectrogram shape
IM_SIZE = (256, 128)

# Resize mode, options are:
# 'crop': Crops from center of the image
# 'cropRandom': Crops from random position
# 'squeeze': Ignores aspect ratio when resizing
# 'fill': Fills with random noise to keep aspect ratio
RESIZE_MODE = 'squeeze'

# Normalization mode (values between -1 and 1)
ZERO_CENTERED_NORMALIZATION = True

# List of rejected specs, which we want to use as noise samples during augmentation
if os.path.exists(NOISE_PATH):
    NOISE_SAMPLES = [os.path.join(NOISE_PATH, s) for s in os.listdir(NOISE_PATH)]
else:
    NOISE_SAMPLES = []

# Image augmentation, uncomment to use; specify mode + value
IM_AUGMENTATION = {#'roll_h':0.5,                   # Horizontal roll
                   'roll_v':0.1,                    # Vertical roll
                   #'crop':[0.1, 0.0, 0.05, 0.0],   # Random crop - top, left, bottom, right
                   #'noise':0.05,                   # Gaussian noise
                   'add':NOISE_SAMPLES,             # List of specs to add to original sample
                   #'brightness':0.15,              # Adjust brightness
                   #'dropout':0.25,                 # Dropout single pixels
                   #'blackout':0.10,                # Dropout entire regions
                   #'blur':3,                       # Image blur
                   #'zoom':0.25,                    # Random zoom (equally cropping each side)
                   #'rotate':10,                    # Rotate by angle
                   #'multiply':0.25,                # Multiply pixel values
                   #'mean':True                     # Substract mean from image
                  }

# Maximum number of random augmentations per image
# Each try has 50% chance of success; we do not use duplicate augmentations
AUGMENTATION_COUNT = 2

# Probability for image augmentation
AUGMENTATION_PROBABILITY = 0.5

#########################  MODEL  #########################

# Changing model settings can have great impact on both, training time and accuracy
# We are using a custom architecture with only a few layers and no shortcuts
# You can find more Lasagne model implementations here: https://github.com/Lasagne/Recipes/tree/master/modelzoo

# Options are: relu, lrelu (leaky relu), vlrelu (very leaky relu), elu and identity
NONLINEARITY = 'relu'

# Number of filters in each convolutional layer group
# You can change the number of groups by changing the amount of
# values in the array (adjust KERNEL_SIZES accordingly!)
# 5 values == 5 convolutional groups
FILTERS = [64, 128, 256, 512, 1024]

# Size of kernels in each convolution (we use 'same' padding)
KERNEL_SIZES = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

NUM_OF_GROUPS = [1, 1, 1, 1, 1]

# Activate Batch Norm
BATCH_NORM = True

# Reduce spatial dimension with MaxPooling (True) or strided convolutions (False)
MAX_POOLING = True

# Specify the type of dropout
# 'random': Standard dropout of random pixels per channel
# 'location': Dropout same pixels across all channels
# 'channel': Dropout of entire channels
DROPOUT_TYPE = 'random'

# Dropout probability (higher == more regularization)
DROPOUT = 0.0

#######################  MODEL I/O ########################

# Name of current run is used as filename
RUN_NAME = 'BirdCLEF_TUC_CLO_EXAMPLE'

# Snapshot directory
MODEL_PATH = 'snapshots/'

# Filename of .pkl-file to load pre-trained model from (default = None, has to be the 'model_params'-file)
PRETRAINED_MODEL_NAME = None # e.g. 'BirdCLEF_TUC_CLO_EXAMPLE_model_params_epoch_50.pkl'

# If the output size of the pre-trained model differs from the current model, set flag to False
LOAD_OUTPUT_LAYER = False

#######################  TRAINING  ########################

# Number of epochs to train
EPOCHS = 100

# Start epoch, important if you use a pre-trained model to continue training
EPOCH_START = 1

# Batch size to use
BATCH_SIZE = 32

# Set learning rate and schedule
LEARNING_RATE = {'start':0.001, 'end':0.000001}

# Options are 'step', 'linear', 'cosine', 'root', 'constant'
# If you want to use steps, write 'step-3' for three steps
# during training to go from start to end lr
LR_SCHEDULE = 'cosine' 

# Impact of L2 measure on loss
L2_WEIGHT = 0

# Optimizer options are: 'adam', 'sgd' and 'nesterov'
OPTIMIZER = 'adam'

# Epochs between snapshot save
SNAPSHOT_EPOCHS = 5

# Epochs to wait before early stopping
EARLY_STOPPING_WAIT = 10

########################  TESTING  ########################

# .pkl file of model to test (not the params-file)
TEST_MODEL = None # e.g. 'BirdCLEF_TUC_CLO_EXAMPLE_model_epoch_50.pkl'

# Maximum amount of randomly selected files from the local validation set (None = no limit)
MAX_TEST_FILES = None

# Limit the amount of test files per class
MAX_TEST_SAMPLES_PER_CLASS = -1

# Limit the amount of (randomly) extracted specs per file (GPU memory!)
MAX_SPECS_PER_FILE = 64

# Include background species in metric (labels need to be sci-names)
TEST_WITH_BG_SPECIES = True

####################  STATS AND LOG  ######################

# Global vars
STATS = {}
DO_BREAK = False

# Options for log mode are 'all', 'info', 'progress', 'error', 'result'
LOG_MODE = 'all'

# Path for final log-file
LOG_FILE = 'BirdCLEF_Logfile.txt'

