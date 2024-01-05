# IGUANe harmonization

<img align='right' src="iguane.png" width="250">

This repository provides code to use the IGUANe model for harmonization of MR images. The full method as well as validation experiments are detailled in the paper (TODO: arxiv link). The model has been trained for harmonization of T1-weighted (T1w) brain images.


## Table of contents

- [Installation instructions](#installation-instructions)
  - [Anaconda environment](#anaconda-environment)
  - [Preprocessing tools](#preprocessing-tools)
- [All-in-one](#all-in-one)
- [Preprocessing](#preprocessing)
- [Harmonization inference](#harmonization-inference)
- [Harmonization training](#harmonization-training)
- [Prediction models](#prediction-models)
- [Metadata](#metadata)


## Installation instructions

IGUANe can be used both with and without GPU. Nevertheless, it is faster with GPU, especially the *HD-BET* brain extraction.

### Anaconda environment

For IGUANe harmonization, you can use the file *./iguane.yml* to create the *iguane* Anaconda environment: `conda env create -f ./iguane.yml`.

To use a GPU, you have to set environment variables (to do before every usage after having activated your environment):
```
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib
```

In our case, we had some problems with libdevice and had to create a symbolic link:
```
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
ln -s $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice
```


### Preprocessing tools

Preprocessing requires additional tools:
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
- [ANTs](http://stnava.github.io/ANTs/)
- [HD-BET](https://github.com/NeuroAI-HD/HD-BET/tree/master), we recommend to install it in the *iguane* environment you previously created by following these steps:
  1. If you use a GPU (recommended if available), install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally), e.g. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  2. Install HD-BET: `git clone https://github.com/MIC-DKFZ/HD-BET && cd HD-BET && pip install -e .`


## All-in-one

The full inference pipeline including preprocessing and harmonization using our trained model can be executed in a straightforward way using the script *./all_in_one.py*.

Todo : décrire l'utilisation du script -> argmuments, GPU, dimensions. Expliquer que c'est mieux de passer un CSV plus rapide. Il faudra aussi modifier le script foireux pour le cropping.