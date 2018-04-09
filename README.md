# Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System
By [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Thomas Wilhelm-Stein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Holger Klinck](http://www.birds.cornell.edu/page.aspx?pid=1735&id=489), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en), and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)

## Introduction
We provide a baseline system for the LifeCLEF bird identification task BirdCLEF2017. We encourage participants to build upon the code base and share their results for future reference. We will keep the repository updated and will add improvements and submission boilerplate in the future. 

<b>If you have any questions or problems running the scripts, don't hesitate to contact us.</b>

Contact:  [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

## Installation
This is a Thenao/Lasagne implementation in Python 2.7 for the identification of hundreds of bird species based on their vocalizations. This code is tested using Ubuntu 16.04 LTS but should work with other distributions as well.

Before cloning the repository, you need to install CUDA, cuDNN, OpenCV, Libav, Theano and Lasagne. You can find more detailed instructions below. After that, you can use the Python package tool PIP to install missing dependencies after the download of the repository:

```
git clone https://github.com/kahst/BirdCLEF-Baseline.git
cd BirdCLEF-Baseline
sudo pip install –r requirements.txt
```

## Dataset
You can download the BirdCLEF training and test data via https://www.crowdai.org. 

You need to register for the challenges to access the data. After download, you need to unpack the archive and change the path to the resulting directory containing "wav" and "xml" folders in the `config.py` script.

## Workflow

Our workflow consists of four main phases: First, we need to <b>sort the BirdCLEF training data</b>. Secondly, we <b>extract spectrograms</b> from audio recordings. Thirdly, we <b>train a deep neural net</b> based on the resulting spectrograms - we treat the audio classification task as an image processing problem. Finally, we <b>test the trained net</b> given a local validation set of unseen audio recordings.

### Sorting the data

We want to divide the dataset into a train and validation split. The validation split should comprise 10% of the entire dataset and should contain at least one sample per bird species. Additionally, we want to copy the samples into folders named after the class they represent. The training script uses subfolders as class names (labels), so the sorted dataset should look like this:

```
dataset
¦
+--train   
¦  ¦
¦  +---species1
¦  ¦      file011.wav
¦  ¦      file012.wav
¦  ¦      ...
¦  ¦   
¦  +---species2
¦  ¦      file021.wav
¦  ¦      file022.wav
¦  ¦      ...
¦  ¦    
¦  +---...
¦
+--val
¦  ¦
¦  +---species1
¦  ¦      file013.wav
¦  ¦      ...
¦  ¦
¦  +---species2
¦  ¦      file023.wav
¦  ¦      ...
¦  ¦
¦  +---...
¦
+--metadata
      file011.json
      file012.json
      ...

```
Before running the script `sort_data.py`, you need to adjust the path pointing to the extracted wav and xml files from the BirdCLEF training data in the `config.py` by setting the value for `TRAINSET_PATH`. We are using the scientific name of each species as label, that makes ist easier to include background species in the metric for evaluation. However, you can use any class name you want - the class ID provided with the xml files would be equally good.

The `metadata` directory contains JSON-files which store some additional information, most importantly the list of background species of each recording.

<i><b>Note:</b> You can use any other dataset for training, as long you organize it in the same way. Simply adjust the sorting script accordingly.</i>

### Spectrogram Extraction

Extracting spectrograms from audio recordings is a vital part of our system. We decided to use MEL-scale log-amplitude spectrograms which each represent one second of a recording. We are using <b>librosa</b> for all of the audio processing. The script `utils/audio.py` contains all the logic. You can run the script stand-alone with the provided example wav-file.

The `config.py` contains a section with all important settings, like sample rate, chunk length and cut-off frequencies. These are the settings we are using as defaults:

```
SAMPLE_RATE = 44100
SPEC_FMIN = 500
SPEC_FMAX = 15000
SPEC_LENGTH = 1.0
SPEC_OVERLAP = 0.25
SPEC_MINLEN = 1.0
SPEC_SIGNAL_THRESHOLD = 0.001
```

Most monophonic recordings from the BirdCLEF dataset are sampled at `44.1 kHz`, we use a low-pass and high-pass filter at `15 kHz` and `500 Hz`. Our signal chunks are of `1 s` length - you can use any other chunk length if you like. The `SPEC_OVERLAP` value defines the step width for extraction, consecutive spectrograms are overlapping by the defined amount. The `SPEC_MINLEN` value excludes all chunks shorter than `1 s` from the extraction.

Our rule-based spectrogram analysis rejects samples which do not contain any bird sounds. It also estimates the signal-to-noise ratio based on some simple calculations. The rejection threshold is set through the `SPEC_SIGNAL_THRESHOLD` value and will be preserved in the filename of the saved spectrogram file.

### Training

...

### Testing

...

## Installation Details

The versions that you need for your machine differ, depending on OS and GPU. The installation process listed below should work with Ubuntu 16.04 LTS and any CUDA-capable GPU by NVIDIA.

### CUDA

Download CUDA 9.1 (you might want to use newer versions, if available): 

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork

Install CUDA:

```
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

Add to environment path:

```
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### cuDNN

Download cuDNN (you need to be registered):

http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

Installing from a Tar File:

Navigate to your <cudnnpath> directory containing the cuDNN Tar file.
Unzip the cuDNN package.

```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
```

Copy the following files into the CUDA Toolkit directory.

```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

### Theano

Prerequisites (incl. Python):

```
sudo apt-get install python-dev
sudo apt-get install python-pip
sudo pip install numpy
sudo pip install scipy
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install cmake
sudo pip install cython
```
Install gpuarray:

http://deeplearning.net/software/libgpuarray/installation.html

```
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
make install
cd ..

python setup.py build
sudo python setup.py install

python setup.py build_ext -L $MY_PREFIX/lib -I $MY_PREFIX/include
sudo ldconfig
```

Install Theano:

```
git clone git://github.com/Theano/Theano.git
cd Theano
sudo pip install -e .
```

### .theanorc

Adjust .theanorc in your home directory to select a GPU and fix random seeds:

```
[global]
device=cuda
floatX=float32
optimizer_excluding=low_memory

[mode]=FAST_RUN

[dnn.conv]
algo_bwd_filter=deterministic
algo_bwd_data=deterministic

[gpuarray]
preallocate=0
```

### Lasagne

Clone the repository and install Lasagne:

```
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

### OpenCV

We use OpenCV for image processing; you can install the cv2 package for Python running this command:

```
sudo apt-get install python-opencv
```

### Libav

The audio processing library Librosa uses the Libav tools:

```
sudo apt-get install libav-tools
```

If you have trouble with some of the installation steps, you can open an issue or contact us. Thenao and Lasagne offer comprehensive installation guides, too - you should consult them for further information.



