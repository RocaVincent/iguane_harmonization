from model_architectures import Generator
import nibabel as nib
from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
import numpy as np


##########INPUTS##############
mri_paths = ['/NAS/coolio/vincent/data/ixi/mni152/n4brains160_normMed/42.nii.gz', '/NAS/coolio/vincent/data/ixi/mni152/n4brains160_normMed/504.nii.gz']
dest_paths = ['/home/vincent/out_images/42.nii.gz','/home/vincent/out_images/504.nii.gz']
weights_path = './iguane_weights.h5'
#############

if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")


gen = Generator()
gen.load_weights(weights_path)


for i,(mri_path,dest_path) in enumerate(zip(mri_paths,dest_paths), start=1):
    print(f"Processing MR image {i}/{len(mri_paths)}", end='\r')
    mri = nib.load(mri_path)
    data = mri.get_fdata()
    mask = data>0
    data = data/500 - 1
    data[~mask] = 0
    data = np.expand_dims(data, axis=(0,4))
    t = convert_to_tensor(data, dtype='float32')
    data = gen(t, training=False).numpy().squeeze()
    data = (data+1) * 500
    data[~mask] = 0
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    nib.save(mri, dest_path)
    