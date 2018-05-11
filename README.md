# Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System
By [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Thomas Wilhelm-Stein](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en), [Holger Klinck](http://www.birds.cornell.edu/page.aspx?pid=1735&id=489), [Danny Kowerko](https://www.tu-chemnitz.de/informatik/mc/staff.php.en), and [Maximilian Eibl](https://www.tu-chemnitz.de/informatik/HomePages/Medieninformatik/team.php.en)

## Introduction
We provide a baseline system for the LifeCLEF bird identification task BirdCLEF2018. We encourage participants to build upon the code base and share their results for future reference. We will keep the repository updated and will add improvements and submission boilerplate in the future. 

<b>If you have any questions or problems running the scripts, don't hesitate to contact us.</b>

Contact:  [Stefan Kahl](http://medien.informatik.tu-chemnitz.de/skahl/about/), [Technische Universität Chemnitz](https://www.tu-chemnitz.de/index.html.en), [Media Informatics](https://www.tu-chemnitz.de/informatik/Medieninformatik/index.php.en)

E-Mail: stefan.kahl@informatik.tu-chemnitz.de

This project is licensed under the terms of the MIT license.

## Citation

Please cite the paper in your publications if the repository helps your research.

```
@article{kahl2018recognizing,
  title={Recognizing Birds from Sound - The 2018 BirdCLEF Baseline System},
  author={Kahl, Stefan and Wilhelm-Stein, Thomas and Klinck, Holger and Kowerko, Danny and Eibl, Maximilian},
  journal={arXiv preprint arXiv:1804.07177},
  year={2018}
}
```

<b>You can download our paper here:</b> [https://arxiv.org/abs/1804.07177](https://arxiv.org/abs/1804.07177)

<i>Score update: We now achive a MLRAP score of 0.61 for a single model and 0.72 for an ensemble on the local validation set with the latest updates using different architectures (e.g. ResNet) and dataset compilations.</i>

## Installation
This is a Thenao/Lasagne implementation in Python 2.7 for the identification of hundreds of bird species based on their vocalizations. This code is tested using Ubuntu 16.04 LTS but should work with other distributions as well.

Before cloning the repository, you need to install CUDA, cuDNN, OpenCV, Libav, Theano and Lasagne. You can find more detailed instructions [below](#installation-details). After that, you can use the Python package tool PIP to install missing dependencies after the download of the repository:

```
git clone https://github.com/kahst/BirdCLEF-Baseline.git
cd BirdCLEF-Baseline
sudo pip install –r requirements.txt
```

## Docker
On your host system you need to ...
1. Install [Docker Engine Utility for NVIDIA GPUs](https://github.com/NVIDIA/nvidia-docker)
2. Clone repository `git clone https://github.com/kahst/BirdCLEF-Baseline.git`
2. Run `./docker-run <path_to_datasets>`

The `docker-run.sh` script takes care of all required tasks (see [Workflow](#workflow))

## Dataset
You can download the BirdCLEF training and test data via https://www.crowdai.org. 

You need to register for the challenges to access the data. After download, you need to unpack the two archives and change the path to the resulting directory containing "wav" and "xml" folders in the `config.py` script.

<i><b>Note:</b> The dataset is quite large, you will need <b>~250 GB</b> for the training data.</i>

## Workflow

Our workflow consists of four main phases: First, we need to <b>sort the BirdCLEF training data</b>. Secondly, we <b>extract spectrograms</b> from audio recordings. Thirdly, we <b>train a deep neural net</b> based on the resulting spectrograms - we treat the audio classification task as an image processing problem. Finally, we <b>test the trained net</b> given a local validation set of unseen audio recordings. We also support the final submission format - you can build a valid submission after training.

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

Extracting spectrograms from audio recordings is a vital part of our system. We decided to use MEL-scale log-amplitude spectrograms, which each represent one second of a recording. We are using <b>librosa</b> for all of the audio processing. The script `utils/audio.py` contains all the logic. You can run the script stand-alone with the provided example wav-file.

You can run the script `spec.py` to start the extraction - this might take a while, depending on your CPU.

The `config.py` contains a section with all important settings, like sample rate, chunk length and cut-off frequencies. We are using these settinsg as defaults:

```
SPEC_TYPE = 'melspec'
SAMPLE_RATE = 44100
SPEC_FMIN = 500
SPEC_FMAX = 15000
SPEC_LENGTH = 1.0
SPEC_OVERLAP = 0.25
SPEC_MINLEN = 1.0
SPEC_SIGNAL_THRESHOLD = 0.001
```

Most monophonic recordings from the BirdCLEF dataset are sampled at `44.1 kHz`, we use a low-pass and high-pass filter at `15 kHz` and `500 Hz`. Our signal chunks are of `1 s` length - you can use any other chunk length if you like. The `SPEC_OVERLAP` value defines the step width for extraction, consecutive spectrograms are overlapping by the defined amount. The `SPEC_MINLEN` value excludes all chunks shorter than `1 s` from the extraction. We support linear and mel-scale spectrograms.

Our rule-based spectrogram analysis rejects samples, which do not contain any bird sounds. It also estimates the signal-to-noise ratio based on some simple calculations. The rejection threshold is set through the `SPEC_SIGNAL_THRESHOLD` value and will be preserved in the filename of the saved spectrogram file.

### Training

If your dataset is sorted and all specs have been extracted, you can start training your own CNN. If you changed some of the paths, make sure to adjust the settings in the `config.py` accordingly.

There are numerous settings that you can change to adjust the net itself and the training process. Most of them might have significant impact on the duration of the training process, memory consumption and result quality.

All options are preceded by a comment explaining the impact of changes - if you still have questions or run into any trouble, please do not hesitate to contact us.

To start the training, simply run the script `train.py`. This will automatically call the following procedures:

- parsing the dataset for samples
- building a neural net
- compiling Thenao test and train functions
- generating batches of samples (incl. augmentation)
- training the net for some epochs
- validating the net after each epoch
- saving snapshots at certain points during training
- saving best snapshopt after training has completed

When finished (this might take a looooong time), you can find the best model in the `snapshot/` directory named after the run name specified in the `config.py`.

<i><b>Note:</b> If you run out of GPU memory, you should consider lowering the batch size and/or input size of the net, or dial down on the parameter count of the net (But hey: Who wants to do that?).</i>

### Testing

We already created a local validation split with `sort_data.py`. We now make use of those unseen recordings and assess the performance of the best snapshot from training (e.g. `TEST_MODEL = 'BirdCLEF_TUC_CLO_EXAMPLE_model_epoch_50.pkl'`). 

Testing includes the spectrogram extraction for each test recording (specify how many specs to use with `MAX_SPECS_PER_FILE`) and the prediction of class scores for each segment. Finally, we calculate the global score for the entire recording by pooling individual scores of all specs. We use <b>Median Filtered Pooling</b> for that - you can change the pooling strategy in the `test.py` by adjusting this lines: 

```
row_median = np.median(p, axis=1, keepdims=True)
p[p < row_median * 1.5] = 0.0
p_pool = np.mean((p * 2) ** 2, axis=0)
p_pool -= p_pool.min()
if p_pool.max() > 1.0:
    p_pool /= p_pool.max()
```

The local validation split from our baseline approach contains 4399 recordings - 10% of the entire training set but at least one recording per species. The metric we use is called <b>Mean Label Ranking Average Precision</b> (MLRAP) and our best net scores a MLRAP of 0.612 including background species (`TEST_WITH_BG_SPECIES = True`).

<b>The results are competitive, but still - there is a lot of room for improvements :)</b>

### Evaluation

If you want to experiment with the system and evaluate different settings or CNN layouts, you can simply change some values in the `config.py` and run the script `evaluate.py`. This will automatically run the training, save a snapshot and test the trained model using the local validation split. All you have to do is sit and wait for a couple of hours :)

### Model Distillation

We support model distillation which allows to compress ('distill') large models or ensembles into smaller models with less parameters. All you need to do is to define a `TEACHER` model (or a list of models) and we will use the teacher predictions as ground truth during training instead of the binary, one-hot targets.

### Submission

You can use our code to build a valid BirdCLEF sumbission for bith tasks - monophone and soundscape. Use the script `submission_monophone.py` and `submission_soundscape.py` after training. You need to specify one or more `TEST_MODELS` and you have to adjust the `TESTSET_PATH` and change it to the individual monophonic and soundscape test paths.

<i><b>Note:</b> You will need to download the test data available from <b>crowdai.org</b>, first.</i>

## Installation Details

The versions that you need for your machine differ, depending on OS and GPU. The installation process listed below should work with Ubuntu 16.04 LTS and any CUDA-capable GPU by NVIDIA.

First of all, you should update your system:

```
sudo apt-get update
sudo apt-get upgrade
```

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

Add the paths to the .bashrc:

```
PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

<i><b>Note:</b> You should be able to run `nvidia-smi` as command and see some details about your GPU. If not, the proper drivers are missing. You can install the drivers for your GPU with e.g. `sudo apt-get install nvidia-390`.</i>

### cuDNN

Download cuDNN (you need to be registered):

https://developer.nvidia.com/cudnn

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
sudo apt-get install python-dev python-pip libblas-dev liblapack-dev cmake
sudo pip install numpy, scipy, cython
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
sudo make install
cd ..

sudo python setup.py build
sudo python setup.py install

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



