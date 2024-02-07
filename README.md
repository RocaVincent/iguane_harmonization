<img align='right' src="iguane.png" width="180">

# IGUANe harmonization

This repository provides code to use the IGUANe model for harmonization of MR images. The full method as well as validation experiments are detailled in an [ArXiv paper](https://arxiv.org/abs/2402.03227). The model has been trained for harmonization of T1-weighted brain images.

Scripts in this repository work on files in Nifti format with **.nii.gz** extension.


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

The full inference pipeline including preprocessing and harmonization using our trained model can be executed in a straightforward way using the script *./all_in_one.py*. There are two ways to run the script:
  - `python all_in_one.py --in-mri <in_mri> --out_mri <out_mri> [options]` -> processes one MR image (input and output filepath)
  - `python all_in_one.py --in-csv <in-csv> [options]` -> processes multiple MR images. The input is a CSV file that has at least two columns: *in_paths* and *out_paths* (input and output filepaths).

The second script will be faster for processing several images. Several supplementary arguments can be specified:
  - `--hd-bet-cpu` -> You must specify this option if you don't have a GPU or don't want to use it for HD-BET brain extraction. Note that the CPU version is slower and is simpler that the GPU one (information [here](https://github.com/MIC-DKFZ/HD-BET/tree/master)).
  - `--n-procs <n>` -> If you have several inputs (*--in-csv* argument), you can specify the number of CPU to use to parallelize the preprocessings (except HD-BET) (default is 1).
  
IGUANe inference will automatically be executed on CPU if no GPU is available.

Note that preprocessing includes image cropping from (182, 218, 182) to (160, 192, 160). For this purpose, the number of background slices on each of the six volume sides is computed, and cropping is performed to ensure that no brain voxels are removed. If there are not enough background slices in one of the three axes, a smaller central crop is taken (176, 208, and 176 for the 1st, 2nd, and 3rd axes, respectively).


## Preprocessing

You can also preprocess the MR images yourself by following these steps:

1. Setting the MR image in the standard MNI152 orientation with [fslreorient2std](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils).
2. Skull-stripping with [HD-BET](https://github.com/NeuroAI-HD/HD-BET).
3. Bias correction with [N4BiasFieldCorrection](https://manpages.ubuntu.com/manpages/trusty/man1/N4BiasFieldCorrection.1.html), using the brain mask computed in step 2.
4. Linear registration with [FSL-FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) towards a MNI152 template (*./preprocessing/MNI152_T1_1mm.nii.gz*) with trilinear interpolation and six degrees of freedom.
5. Normalize the median of the brain intensities to 500. You can use the script *./preprocessing/median_norm.py*. The brain mask associated to each MR image is required. To obtain it, you can apply the transformation computed in step 4 to the brain mask computed in step 2 (with nearestneighbour interpolation).
6. Cropping from `(182,218,182)` dimensions to `(160,192,160)`. You can use the script *./preprocessing/crop_mris.py* for this. Note that this script does not process images with not enough background slices in one the three axes.


## Harmonization inference

To apply IGUANe harmonization separately from preprocessing, you can use the script *./harmonization/inference.py*. Three variables need to be defined:
- `mri_paths`: list of the filepaths of the preprocessed MR images.
- `dest_paths`: list of the destination filepaths for the harmonized MR images.
- `weights_path`: filepath (*.h5*) for the weights of the harmonization model. You can let it to *./iguane_weights.h5* to use the model we trained in our study our use your own model.

The scripts runs faster with GPU but can also be used with CPU only.

You must be in the *./harmonization* directory to use this script.


## Harmonization training

To train your own harmonization model, you can use the scrip *./harmonization/training/main.py*. The following variables need to be defined:
- `dataset_pairs`: A list of infinite iterators corresponding to each source domain. Each one yields a batch with images from the reference domain and images from the source domain. To implement them, you can use one of the two functions defined in *./harmonization/training/input_pipeline/tf_dataset.py*. Both work from files in *TFRecord* format ([documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord)) where each entry must have been encoded from a dictionnary with `mri` key associated with the MR matrix as value. **Important:** The intensities must have been scaled/shifted to have median of brain intensity and backgroud to 0 and -1, respectively (`mri = mri/500-1` after [preprocessing](#preprocessing)).
  - `datasets_from_tfrecords`: Creates dataset pairs without bias sampling.
  - `datasets_from_tfrecords_biasSampling`: Creates dataset pairs with the bias sampling strategy described in our paper. The function we used to compute the sampling probabilities in our experiments is in *./harmonization/training/input_pipeline/bias_sampling_age.py*.
- `DEST_DIR_PATH`: directory path to which the model weights and the training statistics will be saved.
- `N_EPOCHS`, `STEPS_PER_EPOCH`
- `eval_model`: a validation function that takes no arguments and returns a score to maximize. This function is executed every `EVAL_FREQ` epochs and the weights of the best model are saved in *<DEST_DIR_PATH>/best_genUniv.h5*. You can also change the value of `EVAL_FREQ`. 

You can change the image shape in *./harmonization/training/input_pipeline/constants.py*. A GPU needs to be available to run this script.

You must be in the *./harmonization/training* directory to use this script.


## Prediction models

We provide the code we used for our age prediction models and our binary classifiers.

- *./prediction/training/main.py*: script for training; the following variables must be defined:
  - `dataset`: TensorFlow dataset that yields a batch of MR images with corresponding targets. You can use the function defined in *./prediction/training/input_pipeline/tf_dataset.py*.
  - `loss`: loss function (e.g. `'mae'` for regression, `tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)` for classification)
  - `DEST_DIR_PATH`: directory path to which the model weights and the training statistics will be saved
  - `N_EPOCHS`, `STEPS_PER_EPOCH`
- *./prediction/inference.py*: script for training; the following variables must be defined:
  - `mri_paths`
  - `ids_dict`: a dictionary with fields enabling the image identifications in the output CSV (e.g. `{'sub_id':[1,2], 'session':['sesA','sesB']}`)
  - `intensity_norm`: function to apply on each image before inference (e.g. `def intensity_norm(mri): return mri/500 - 1`)
  - `activation`: last activation function (e.g. *sigmoid* for binary classifier, identity for regressor)
  - `model_weights`: filepath (*.h5*) for the weights of the predictive model.
  - `csv_dest`
  - `IMAGE_SHAPE` -> must correspond with the shapes in `mri_paths` (with channel dimension)
  
For training, you can change the image shape in *./prediction/training/input_pipeline/constants.py*. You must be in the *./prediction/training* to run the script. A GPU needs to be available.

For inference, you must be in the *./prediction* directory.


## Metadata

We provide the metadata for the different datasets we used in our study in the *./metadata/* directory.