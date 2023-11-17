# IGUANe harmonization

<img align='right' src="iguane.png" width="250">

This repository provides code to use the IGUANe model for harmonization of MR images. The full method as well as validation experiments are detailled in the paper (TODO: arxiv link). The model has been trained for harmonization of T1-weighted brain images. It can be used in a straightforward for harmonization of your own MR images (see [Harmonization inference](#harmonization-inference)). Harmonization of other types of 3D images can be carried out by retraining a model (see [Harmonization training](#harmonization-training)). For both inference and training, the MR images should be preprocessed by following our pipeline (see [Preprocessing](#Preprocessing)).


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

**Note:** You must be in the *./harmonization* directory to use the script.

## Harmonization training

Tfrecords between min and median at -1 and 0. Dimensions divisible by 16.


Training prediction : cache,shuffle potentiellement à modifier (mémoire).
Training prediction : loss régression ou classif binaire


Explication installation packages (numpy pandas scipy tensorflow).


Les éléments à définir avant de lancer les entraînements. (évoquer IMAGE_SHAPE)


Indiquer les outputs dans les DEST_DIR_PATHS.


Voir pour GPU/CPU. Expliquer qu'avec 10 sites source,on a réussi à entraîner avec GPU 24GB.


Indiquer les types de fichiers d'inputs pour l'inférence (nib.load).
