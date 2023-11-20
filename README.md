# IGUANe harmonization

<img align='right' src="iguane.png" width="250">

This repository provides code to use the IGUANe model for harmonization of MR images. The full method as well as validation experiments are detailled in the paper (TODO: arxiv link). The model has been trained for harmonization of T1-weighted brain images. It can be used in a straightforward way for harmonization of your own MR images (see [Harmonization inference](#harmonization-inference)). Harmonization of other types of 3D images can be carried out by retraining a model (see [Harmonization training](#harmonization-training)). For both inference and training, the MR images should be preprocessed by following our pipeline (see [Preprocessing](#Preprocessing)).


## Preprocessing

To use our trained IGUANe model appropriately, you should follow the following preprocessing pipeline:
1. Setting the MR image in the standard MNI152 orientation with [fslreorient2std](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils).
2. Skull-stripping with [HD-BET](https://github.com/MIC-DKFZ/HD-BET).
3. Bias correction with [N4BiasFieldCorrection](https://manpages.ubuntu.com/manpages/trusty/man1/N4BiasFieldCorrection.1.html), using the brain mask computed in step 2.
4. Linear registration with [FSL-FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) towards a MNI152 template (*./preprocessing/MNI152_T1_1mm.nii.gz*) with trilinear interpolation and six degrees of freedom.
5. Normalize the median of the brain intensities to 500. You can use the script *./preprocessing/median_norm.py*. The brain mask associated to each MR image is required. To obtain it, you can apply the transformation computed in step 4 to the brain mask computed in step 2 (with nearestneighbour interpolation).
6. Cropping from `(182,218,182)` dimensions to `(160,192,160)`. You can use the script *./preprocessing/crop_mris.py* for this.


**Note on image cropping:** The inputs to give to *./preprocessing/crop_mris.py* are explained at the top of the script. When the brain is too large, cropping is not applied. In that specific case, you can still use IGUANe to harmonize the image by making the dimensions divisible by 16 (e.g. `(176,208,176)`).


## Environment setup

Creation of the *iguane* Anaconda environment: `conda create -f ./conda_env.yml`.

Setting environment variables after activation of the environment (to do before every usage):
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

## Harmonization inference

To apply IGUANe harmonization, you can use the script *./harmonization/inference.py*. Three variables need to be defined:
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


## Prediction tasks

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
  
For training, you can change the image shape in *./prediction/training/input_pipeline/constants.py*. You must in the *./prediction/training* to run the script.

For inference, you must be in the *./prediction* directory.


## Metadata

We provide the metadata for the different datasets we used in our study in the *./metadata/* directory.



