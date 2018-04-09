# This file includes basic functionality for audio signal processing
# including audio file handling, signal splitting and spectrogram extraction
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os

import numpy as np
import librosa

import cv2

def openAudioFile(path, sample_rate=44100, as_mono=True, mean_substract=False):
    
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=as_mono)

    # Noise reduction?
    if mean_substract:
        sig -= sig.mean()

    return sig, rate

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))
        
        sig_splits.append(split)

    return sig_splits

def melspec(sig, rate, shape=(128, 256), fmin=500, fmax=15000, normalize=True, preemphasis=0.95):

    # shape = (height, width) in pixels

    # Mel-Spec parameters
    SAMPLE_RATE = rate
    N_FFT = shape[0] * 8 # = window length
    N_MELS = shape[0]
    HOP_LEN = len(sig) // (shape[1] - 1)    
    FMAX = fmax
    FMIN = fmin

    # Preemphasis as in python_speech_features by James Lyons
    if preemphasis:
        sig = np.append(sig[0], sig[1:] - preemphasis * sig[:-1])

    # Librosa mel-spectrum
    melspec = librosa.feature.melspectrogram(y=sig, sr=SAMPLE_RATE, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX, fmin=FMIN, power=2.0)
    
    # Convert power spec to dB scale (compute dB relative to peak power)
    melspec = librosa.power_to_db(melspec, ref_power=np.max, top_db=100)

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    melspec = melspec[::-1, ...]

    # Trim to desired shape if too large
    melspec = melspec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    if normalize:
        melspec -= melspec.min()
        melspec /= melspec.max()

    return melspec.astype('float32')

def signal2noise(spec):

    # Get working copy
    spec = spec.copy()

    # Calculate median for columns and rows
    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)

    # Binary threshold
    spec[spec < row_median * 1.25] = 0.0
    spec[spec < col_median * 1.15] = 0.0
    spec[spec > 0] = 1.0

    # Median blur
    spec = cv2.medianBlur(spec, 3)

    # Morphology
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))

    # Sum of all values
    spec_sum = spec.sum()

    # Signal to noise ratio (higher is better)
    try:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
    except:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1])

    return s2n

def specsFromSignal(sig, rate, shape, seconds, overlap, minlen, **kwargs):

    # Split signal in consecutive chunks with overlap
    sig_splits = splitSignal(sig, rate, seconds, overlap, minlen)

    # Extract specs for every sig split
    for sig in sig_splits:

        # Get spec for signal chunk
        spec = melspec(sig, rate, shape, **kwargs)

        yield spec

def specsFromFile(path, rate, seconds, overlap, minlen, shape, start=-1, end=-1, **kwargs):

    # Open file
    sig, rate = openAudioFile(path, rate)

    # Trim signal?
    if start > -1 and end > -1:
        sig = sig[int(start * rate):int(end * rate)]
        minlen = 0

    # Yield all specs for file
    for mspec in specsFromSignal(sig, rate, shape, seconds, overlap, minlen, **kwargs):
        yield mspec
    
if __name__ == '__main__':

    for spec in specsFromFile('../example/Acadian Flycatcher.wav', 44100, 1, 0, 1, shape=(128, 256), fmin=300, fmax=15000):

        # Show spec and wait for enter key
        cv2.imshow('SPEC', spec)
        cv2.waitKey(-1)

        # Calculate and show noise measure
        noise = signal2noise(spec)
        print noise
