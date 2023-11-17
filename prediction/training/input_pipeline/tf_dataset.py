from .data_augmentation import augmentation
from tensorflow import expand_dims as tf_expand_dims, TensorSpec
from tensorflow.data import AUTOTUNE, Dataset
from .constants import IMAGE_SHAPE
import nibabel as nib


def dataset_from_pathList(mri_paths, targets, intensity_norm, batch_size):
    def python_gen():
        for path,target in zip(mri_paths,targets):
            mri = intensity_norm(nib.load(path).get_fdata())
            yield tf_expand_dims(mri, axis=3), target
            
    ds = Dataset.from_generator(python_gen, output_signature=(TensorSpec(shape=IMAGE_SHAPE, dtype='float32'),
                                                              TensorSpec(shape=(), dtype='float32')))
    ds = ds.cache().shuffle(len(mri_paths))
    ds = ds.map(lambda mri,target: (augmentation(mri), target), num_parallel_calls=AUTOTUNE, deterministic=False)
    return ds.batch(batch_size, drop_remainder=True, num_parallel_calls=AUTOTUNE, deterministic=False).prefetch(AUTOTUNE)